# ASSET_LIBRARY_SPEC.md — Asset Library & IP-Adapter Integration

**Date:** 2026-03-20
**Status:** Ready for implementation
**Depends on:** None (no blocking specs)
**Blocked by:** Nothing — can start immediately

---

## Overview

This spec adds three capabilities to MusicVision:

1. **Asset Library** — A core module (`src/musicvision/assets/`) for creating, storing, and managing reusable project assets (characters, props, locations) with reference images.
2. **IP-Adapter support in FluxEngine** — Extend FLUX image generation to accept IP-Adapter reference images alongside LoRA, enabling zero-shot character/style consistency without training.
3. **Automatic consistency resolution** — When a scene references assets, the system auto-selects the best available conditioning (LoRA, IP-Adapter, both, or prompt-only) based on what's configured per asset and which image engine is active.

LoRA training is **out of scope** for this spec. The asset library stores training datasets (images + captions), but the training panel is a separate spec. This spec provides the data model hooks that the training panel will consume.

---

## Phased Implementation Order

| Phase | Scope | Depends On |
|-------|-------|------------|
| 1 | Data model: `AssetDef`, enums, `StyleSheet` migration | Nothing |
| 2 | Asset storage: directory structure, `AssetService` CRUD | Phase 1 |
| 3 | IP-Adapter in `FluxEngine` | Phase 1 |
| 4 | Consistency resolver: `resolve_scene_conditioning()` | Phase 1–3 |
| 5 | Wire conditioning into CLI + API image generation | Phase 4 |
| 6 | Asset CRUD API endpoints | Phase 2 |
| 7 | Embedding precomputation + caching | Phase 3 |
| 8 | React Asset Library panel | Phase 6 |

Each phase should pass all existing tests before proceeding. Run `pytest tests/ -v --tb=short` and `ruff check src/ tests/` after each phase.

---

## Phase 1 — Data Model

### New Enums

Add to `src/musicvision/models.py`:

```python
class AssetType(str, Enum):
    """Type of reusable project asset."""
    CHARACTER = "character"
    PROP = "prop"
    LOCATION = "location"


class ConsistencyMethod(str, Enum):
    """How an asset's visual identity is enforced during image generation."""
    NONE = "none"                # prompt-only — description injected, no conditioning
    IP_ADAPTER = "ip_adapter"    # reference image → IP-Adapter at inference time
    LORA = "lora"                # trained LoRA weights fused into pipeline
    BOTH = "both"                # LoRA + IP-Adapter (strongest consistency)
```

### New Models

Add to `src/musicvision/models.py`:

```python
class AssetImage(BaseModel):
    """A single image belonging to an asset."""
    filename: str              # relative to the asset's directory (e.g. "reference_01.png")
    role: str = "reference"    # "reference" | "training" — reference for IP-Adapter/display, training for LoRA datasets
    caption: str = ""          # text caption for LoRA training datasets
    is_primary: bool = False   # the image used for IP-Adapter conditioning and thumbnails


class AssetDef(BaseModel):
    """Universal definition for a reusable project asset (character, prop, or location)."""
    id: str                                              # unique within project, e.g. "singer", "guitar", "rooftop"
    name: str                                            # human-friendly display name
    asset_type: AssetType
    description: str = ""                                # injected into generation prompts

    # Visual references
    images: list[AssetImage] = Field(default_factory=list)

    # Consistency configuration
    consistency: ConsistencyMethod = ConsistencyMethod.NONE
    lora_path: Optional[str] = None                      # relative to project root (e.g. "loras/singer.safetensors")
    lora_weight: float = 0.8                             # LoRA fusion scale (0.0–1.0)
    ip_adapter_scale: float = 0.6                        # IP-Adapter influence (0.0–1.0)
    ip_adapter_embedding_path: Optional[str] = None      # cached precomputed embedding (e.g. "ip_cache/singer.ipadpt")

    @property
    def primary_image(self) -> Optional[AssetImage]:
        """Return the primary reference image, or the first image if none is marked primary."""
        for img in self.images:
            if img.is_primary:
                return img
        return self.images[0] if self.images else None

    @property
    def training_images(self) -> list[AssetImage]:
        """Return all images marked for LoRA training."""
        return [img for img in self.images if img.role == "training"]

    @property
    def reference_images(self) -> list[AssetImage]:
        """Return all images marked as references (IP-Adapter / display)."""
        return [img for img in self.images if img.role == "reference"]

    @property
    def has_lora(self) -> bool:
        """True if this asset has a LoRA path and its consistency method uses LoRA."""
        return self.lora_path is not None and self.consistency in (
            ConsistencyMethod.LORA, ConsistencyMethod.BOTH,
        )

    @property
    def has_ip_adapter(self) -> bool:
        """True if this asset has reference images and its consistency method uses IP-Adapter."""
        return bool(self.images) and self.consistency in (
            ConsistencyMethod.IP_ADAPTER, ConsistencyMethod.BOTH,
        )
```

### StyleSheet Changes

Modify the existing `StyleSheet` model:

```python
class StyleSheet(BaseModel):
    concept: str = ""
    visual_style: str = ""
    color_palette: str = ""
    aspect_ratio: str = "16:9"
    resolution: str = "1280x720"

    # --- NEW ---
    assets: list[AssetDef] = Field(default_factory=list)
    style_lora_path: Optional[str] = None    # project-wide style LoRA applied to all generations
    style_lora_weight: float = 0.7

    # --- DEPRECATED — migrated to assets on load ---
    characters: list[CharacterDef] = Field(default_factory=list)
    props: list[PropDef] = Field(default_factory=list)
    settings: list[SettingDef] = Field(default_factory=list)

    # --- Lookup helpers ---
    def get_asset(self, asset_id: str) -> Optional[AssetDef]:
        """Find an asset by ID. Returns None if not found."""
        return next((a for a in self.assets if a.id == asset_id), None)

    def assets_by_type(self, asset_type: AssetType) -> list[AssetDef]:
        """Return all assets of a given type."""
        return [a for a in self.assets if a.asset_type == asset_type]
```

### Migration Validator

Add a `model_validator` to `StyleSheet` that auto-migrates old `characters`/`props`/`settings` into `assets`:

```python
@model_validator(mode="before")
@classmethod
def _migrate_legacy_assets(cls, data):
    """Migrate old CharacterDef/PropDef/SettingDef into unified assets list."""
    if not isinstance(data, dict):
        return data

    assets = list(data.get("assets", []))
    existing_ids = {a["id"] if isinstance(a, dict) else a.id for a in assets}

    # Migrate characters
    for char in data.get("characters", []):
        c = char if isinstance(char, dict) else char.model_dump()
        if c["id"] not in existing_ids:
            asset = {
                "id": c["id"],
                "name": c.get("name", c["id"]),
                "asset_type": "character",
                "description": c.get("description", ""),
                "images": [],
                "consistency": "none",
                "lora_path": c.get("lora_path"),
                "lora_weight": c.get("lora_weight", 0.8),
            }
            # If it had a reference_image, add it
            if c.get("reference_image"):
                asset["images"] = [{"filename": c["reference_image"], "role": "reference", "is_primary": True}]
            # If it had a lora_path, set consistency
            if c.get("lora_path"):
                asset["consistency"] = "lora"
            assets.append(asset)
            existing_ids.add(c["id"])

    # Migrate props
    for prop in data.get("props", []):
        p = prop if isinstance(prop, dict) else prop.model_dump()
        if p["id"] not in existing_ids:
            asset = {
                "id": p["id"],
                "name": p.get("name", p["id"]),
                "asset_type": "prop",
                "description": p.get("description", ""),
                "images": [],
                "consistency": "none",
            }
            if p.get("reference_image"):
                asset["images"] = [{"filename": p["reference_image"], "role": "reference", "is_primary": True}]
            assets.append(asset)
            existing_ids.add(p["id"])

    # Migrate settings → locations
    for setting in data.get("settings", []):
        s = setting if isinstance(setting, dict) else setting.model_dump()
        if s["id"] not in existing_ids:
            asset = {
                "id": s["id"],
                "name": s.get("name", s["id"]),
                "asset_type": "location",
                "description": s.get("description", ""),
                "images": [],
                "consistency": "none",
            }
            if s.get("reference_image"):
                asset["images"] = [{"filename": s["reference_image"], "role": "reference", "is_primary": True}]
            assets.append(asset)
            existing_ids.add(s["id"])

    data["assets"] = assets
    return data
```

### ImageGenConfig Changes

Add IP-Adapter configuration to `ImageGenConfig`:

```python
class IPAdapterConfig(BaseModel):
    """IP-Adapter configuration for FLUX engines."""
    enabled: bool = False
    model_repo: str = "XLabs-AI/flux-ip-adapter"          # HuggingFace repo for adapter weights
    weight_name: str = "ip_adapter.safetensors"            # filename within repo
    image_encoder_repo: str = "openai/clip-vit-large-patch14"  # CLIP vision encoder
    default_scale: float = 0.6                             # default influence scale

class ImageGenConfig(BaseModel):
    model: ImageModel = ImageModel.FLUX_DEV
    quant: FluxQuant = FluxQuant.AUTO
    steps: Optional[int] = None
    guidance_scale: float = 3.5
    lora_path: Optional[str] = None
    lora_weight: float = 0.8
    ip_adapter: IPAdapterConfig = Field(default_factory=IPAdapterConfig)  # NEW

    @property
    def effective_steps(self) -> int:
        if self.steps is not None:
            return self.steps
        return 4 if self.model == ImageModel.FLUX_SCHNELL else 28
```

### Frontend Type Changes

Update `frontend/src/api/types.ts`:

```typescript
export type AssetType = "character" | "prop" | "location";
export type ConsistencyMethod = "none" | "ip_adapter" | "lora" | "both";

export interface AssetImage {
  filename: string;
  role: string;        // "reference" | "training"
  caption: string;
  is_primary: boolean;
}

export interface AssetDef {
  id: string;
  name: string;
  asset_type: AssetType;
  description: string;
  images: AssetImage[];
  consistency: ConsistencyMethod;
  lora_path: string | null;
  lora_weight: number;
  ip_adapter_scale: number;
  ip_adapter_embedding_path: string | null;
}

// Update StyleSheet to include assets
export interface StyleSheet {
  concept: string;
  visual_style: string;
  color_palette: string;
  aspect_ratio: string;
  resolution: string;
  assets: AssetDef[];
  style_lora_path: string | null;
  style_lora_weight: number;
}
```

### Tests — Phase 1

```python
# tests/test_asset_model.py

class TestAssetDef:
    def test_primary_image_explicit(self):
        asset = AssetDef(
            id="singer", name="Singer", asset_type=AssetType.CHARACTER,
            images=[
                AssetImage(filename="a.png"),
                AssetImage(filename="b.png", is_primary=True),
            ],
        )
        assert asset.primary_image.filename == "b.png"

    def test_primary_image_fallback_to_first(self):
        asset = AssetDef(
            id="singer", name="Singer", asset_type=AssetType.CHARACTER,
            images=[AssetImage(filename="a.png")],
        )
        assert asset.primary_image.filename == "a.png"

    def test_primary_image_none_when_empty(self):
        asset = AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER)
        assert asset.primary_image is None

    def test_has_lora(self):
        asset = AssetDef(
            id="singer", name="Singer", asset_type=AssetType.CHARACTER,
            lora_path="loras/singer.safetensors", consistency=ConsistencyMethod.LORA,
        )
        assert asset.has_lora
        assert not asset.has_ip_adapter

    def test_has_ip_adapter(self):
        asset = AssetDef(
            id="sidekick", name="Sidekick", asset_type=AssetType.CHARACTER,
            consistency=ConsistencyMethod.IP_ADAPTER,
            images=[AssetImage(filename="ref.png", is_primary=True)],
        )
        assert asset.has_ip_adapter
        assert not asset.has_lora

    def test_has_both(self):
        asset = AssetDef(
            id="hero", name="Hero", asset_type=AssetType.CHARACTER,
            consistency=ConsistencyMethod.BOTH,
            lora_path="loras/hero.safetensors",
            images=[AssetImage(filename="ref.png", is_primary=True)],
        )
        assert asset.has_lora
        assert asset.has_ip_adapter

    def test_training_images_filter(self):
        asset = AssetDef(
            id="singer", name="Singer", asset_type=AssetType.CHARACTER,
            images=[
                AssetImage(filename="ref.png", role="reference"),
                AssetImage(filename="train_01.png", role="training", caption="Singer on stage"),
                AssetImage(filename="train_02.png", role="training", caption="Singer closeup"),
            ],
        )
        assert len(asset.training_images) == 2
        assert len(asset.reference_images) == 1


class TestStyleSheetMigration:
    def test_legacy_characters_migrate(self):
        data = {
            "visual_style": "cinematic",
            "characters": [
                {"id": "singer", "description": "Woman with red hair", "lora_path": "loras/singer.safetensors"}
            ],
        }
        ss = StyleSheet.model_validate(data)
        assert len(ss.assets) == 1
        assert ss.assets[0].id == "singer"
        assert ss.assets[0].asset_type == AssetType.CHARACTER
        assert ss.assets[0].lora_path == "loras/singer.safetensors"
        assert ss.assets[0].consistency == ConsistencyMethod.LORA

    def test_legacy_props_migrate(self):
        data = {
            "props": [{"id": "guitar", "description": "Red electric guitar", "reference_image": "props/guitar.png"}],
        }
        ss = StyleSheet.model_validate(data)
        assert len(ss.assets) == 1
        assert ss.assets[0].asset_type == AssetType.PROP
        assert ss.assets[0].images[0].filename == "props/guitar.png"

    def test_legacy_settings_migrate_to_locations(self):
        data = {
            "settings": [{"id": "rooftop", "description": "City rooftop at sunset"}],
        }
        ss = StyleSheet.model_validate(data)
        assert ss.assets[0].asset_type == AssetType.LOCATION

    def test_no_duplicate_migration(self):
        """If asset already exists in assets list, don't re-migrate from legacy."""
        data = {
            "assets": [{"id": "singer", "name": "Singer", "asset_type": "character", "description": "Updated"}],
            "characters": [{"id": "singer", "description": "Old description"}],
        }
        ss = StyleSheet.model_validate(data)
        assert len(ss.assets) == 1
        assert ss.assets[0].description == "Updated"

    def test_existing_projects_load_unchanged(self):
        """Projects with no legacy fields and no assets load cleanly."""
        data = {"visual_style": "cinematic", "color_palette": "warm tones"}
        ss = StyleSheet.model_validate(data)
        assert ss.assets == []

    def test_roundtrip_with_assets(self, tmp_path):
        config = ProjectConfig(
            name="Test",
            style_sheet=StyleSheet(
                visual_style="cinematic",
                assets=[
                    AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER,
                             description="Woman with red hair", consistency=ConsistencyMethod.LORA,
                             lora_path="loras/singer.safetensors"),
                ],
            ),
        )
        path = tmp_path / "project.yaml"
        config.save(path)
        loaded = ProjectConfig.load(path)
        assert len(loaded.style_sheet.assets) == 1
        assert loaded.style_sheet.assets[0].has_lora
```

### Backward Compatibility

- `Scene.characters`, `Scene.props`, `Scene.settings` fields continue to store asset IDs as `list[str]`. No change.
- Old `project.yaml` files with `characters`/`props`/`settings` under `style_sheet` auto-migrate on load.
- Old `CharacterDef`, `PropDef`, `SettingDef` classes remain in `models.py` (deprecated but not removed) so existing code that references them doesn't break during migration.
- The migration validator runs before Pydantic validation, so no explicit migration step is needed.

---

## Phase 2 — Asset Storage & Service

### Directory Structure

```
<project_root>/
├── assets/
│   ├── characters/
│   │   ├── singer/
│   │   │   ├── reference_01.png      # IP-Adapter / display
│   │   │   ├── reference_02.png
│   │   │   └── training/             # LoRA training dataset
│   │   │       ├── 001.png
│   │   │       ├── 001.txt           # caption file
│   │   │       ├── 002.png
│   │   │       └── 002.txt
│   │   └── sidekick/
│   │       └── reference_01.png
│   ├── props/
│   │   └── guitar/
│   │       └── reference_01.png
│   └── locations/
│       └── rooftop/
│           └── reference_01.png
├── loras/                            # trained LoRA output directory
│   ├── singer.safetensors
│   └── style.safetensors
├── ip_cache/                         # precomputed IP-Adapter embeddings
│   ├── singer.ipadpt
│   └── sidekick.ipadpt
└── ...
```

### ProjectPaths Additions

Add to `ProjectPaths` (in `project.py` or wherever it lives):

```python
@property
def assets_dir(self) -> Path:
    return self.root / "assets"

def asset_dir(self, asset_type: AssetType, asset_id: str) -> Path:
    """Directory for a specific asset's files."""
    type_dir = {"character": "characters", "prop": "props", "location": "locations"}
    return self.assets_dir / type_dir[asset_type] / asset_id

def asset_training_dir(self, asset_type: AssetType, asset_id: str) -> Path:
    return self.asset_dir(asset_type, asset_id) / "training"

@property
def loras_dir(self) -> Path:
    return self.root / "loras"

@property
def ip_cache_dir(self) -> Path:
    return self.root / "ip_cache"
```

### AssetService

Create `src/musicvision/assets/service.py`:

```python
class AssetService:
    """Manages asset lifecycle: create, update, delete, image upload."""

    def __init__(self, project: ProjectService):
        self.project = project

    def list_assets(self, asset_type: Optional[AssetType] = None) -> list[AssetDef]:
        """List all assets, optionally filtered by type."""
        assets = self.project.config.style_sheet.assets
        if asset_type:
            return [a for a in assets if a.asset_type == asset_type]
        return list(assets)

    def get_asset(self, asset_id: str) -> Optional[AssetDef]:
        return self.project.config.style_sheet.get_asset(asset_id)

    def create_asset(self, asset_id: str, name: str, asset_type: AssetType, description: str = "") -> AssetDef:
        """Create a new asset. Creates the directory structure. Saves config."""
        if self.get_asset(asset_id):
            raise ValueError(f"Asset '{asset_id}' already exists")

        asset = AssetDef(id=asset_id, name=name, asset_type=asset_type, description=description)
        self.project.config.style_sheet.assets.append(asset)

        # Create directory
        asset_dir = self.project.paths.asset_dir(asset_type, asset_id)
        asset_dir.mkdir(parents=True, exist_ok=True)

        self.project.save_config()
        return asset

    def update_asset(self, asset_id: str, **updates) -> AssetDef:
        """Update asset fields. Saves config."""
        asset = self.get_asset(asset_id)
        if not asset:
            raise ValueError(f"Asset '{asset_id}' not found")

        for key, value in updates.items():
            if hasattr(asset, key):
                setattr(asset, key, value)

        self.project.save_config()
        return asset

    def delete_asset(self, asset_id: str) -> None:
        """Delete asset, its directory, and cached embeddings. Saves config."""
        asset = self.get_asset(asset_id)
        if not asset:
            raise ValueError(f"Asset '{asset_id}' not found")

        # Remove from config
        self.project.config.style_sheet.assets = [
            a for a in self.project.config.style_sheet.assets if a.id != asset_id
        ]

        # Remove directory
        asset_dir = self.project.paths.asset_dir(asset.asset_type, asset_id)
        if asset_dir.exists():
            import shutil
            shutil.rmtree(asset_dir)

        # Remove cached embedding
        if asset.ip_adapter_embedding_path:
            cache_path = self.project.resolve_path(asset.ip_adapter_embedding_path)
            if cache_path.exists():
                cache_path.unlink()

        self.project.save_config()

    def add_image(
        self,
        asset_id: str,
        source_path: Path,
        role: str = "reference",
        caption: str = "",
        is_primary: bool = False,
    ) -> AssetImage:
        """Copy an image into the asset's directory and register it."""
        asset = self.get_asset(asset_id)
        if not asset:
            raise ValueError(f"Asset '{asset_id}' not found")

        # Determine destination
        if role == "training":
            dest_dir = self.project.paths.asset_training_dir(asset.asset_type, asset_id)
        else:
            dest_dir = self.project.paths.asset_dir(asset.asset_type, asset_id)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy file
        dest = dest_dir / source_path.name
        if dest.exists():
            # Auto-suffix to avoid overwrite
            stem = source_path.stem
            suffix = source_path.suffix
            counter = 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{counter:02d}{suffix}"
                counter += 1
        import shutil
        shutil.copy2(source_path, dest)

        # Make path relative to project root
        rel_path = str(dest.relative_to(self.project.paths.root))

        # If setting as primary, unset existing primary
        if is_primary:
            for img in asset.images:
                img.is_primary = False

        # If first image and no primary set, auto-primary
        if not asset.images:
            is_primary = True

        img = AssetImage(filename=rel_path, role=role, caption=caption, is_primary=is_primary)
        asset.images.append(img)

        # Save caption file alongside training images
        if role == "training" and caption:
            caption_path = dest.with_suffix(".txt")
            caption_path.write_text(caption, encoding="utf-8")

        self.project.save_config()
        return img

    def remove_image(self, asset_id: str, filename: str) -> None:
        """Remove an image from an asset. Deletes the file."""
        asset = self.get_asset(asset_id)
        if not asset:
            raise ValueError(f"Asset '{asset_id}' not found")

        img = next((i for i in asset.images if i.filename == filename), None)
        if not img:
            raise ValueError(f"Image '{filename}' not found on asset '{asset_id}'")

        # Delete file
        full_path = self.project.resolve_path(filename)
        if full_path.exists():
            full_path.unlink()
            # Also delete caption file for training images
            caption_path = full_path.with_suffix(".txt")
            if caption_path.exists():
                caption_path.unlink()

        asset.images = [i for i in asset.images if i.filename != filename]

        # If we removed the primary, promote the first remaining image
        if img.is_primary and asset.images:
            asset.images[0].is_primary = True

        self.project.save_config()

    def set_primary_image(self, asset_id: str, filename: str) -> None:
        """Set which image is the primary reference."""
        asset = self.get_asset(asset_id)
        if not asset:
            raise ValueError(f"Asset '{asset_id}' not found")

        for img in asset.images:
            img.is_primary = (img.filename == filename)

        self.project.save_config()

    def invalidate_embedding_cache(self, asset_id: str) -> None:
        """Delete cached IP-Adapter embedding when reference images change."""
        asset = self.get_asset(asset_id)
        if asset and asset.ip_adapter_embedding_path:
            cache_path = self.project.resolve_path(asset.ip_adapter_embedding_path)
            if cache_path.exists():
                cache_path.unlink()
            asset.ip_adapter_embedding_path = None
            self.project.save_config()
```

---

## Phase 3 — IP-Adapter in FluxEngine

### FluxEngine.load() Changes

When `config.ip_adapter.enabled` is True, load the IP-Adapter after loading the base pipeline:

```python
def load(self) -> None:
    # ... existing pipeline loading code ...

    # Load project-level LoRA if configured
    if self.config.lora_path:
        self._apply_lora(self.config.lora_path, self.config.lora_weight)

    # --- NEW: Load IP-Adapter ---
    if self.config.ip_adapter.enabled:
        self._load_ip_adapter()

    log.info("FLUX engine ready (%s, %s)", self.config.model.value, strategy)


def _load_ip_adapter(self) -> None:
    """Load IP-Adapter weights into the pipeline."""
    ipa = self.config.ip_adapter
    log.info("Loading IP-Adapter from %s", ipa.model_repo)
    self._pipe.load_ip_adapter(
        ipa.model_repo,
        weight_name=ipa.weight_name,
        image_encoder_pretrained_model_name_or_path=ipa.image_encoder_repo,
    )
    self._ip_adapter_loaded = True
    log.info("IP-Adapter loaded (encoder: %s)", ipa.image_encoder_repo)
```

**VRAM note:** The CLIP-ViT-L image encoder adds ~1.5 GB. On the RTX 5090 (32 GB) this is trivial. On smaller GPUs, the encoder can be offloaded to CPU after embedding precomputation. The `enable_model_cpu_offload()` call that already exists in Tier B/C/D strategies handles this automatically — the image encoder gets offloaded along with everything else.

**Important ordering:** `load_ip_adapter()` must be called AFTER any `enable_model_cpu_offload()` or `enable_sequential_cpu_offload()` calls. If called before, the image encoder is not included in the offload graph and will error. The current `FluxEngine.load()` applies offload during `_load_*()` helpers and LoRA after — IP-Adapter goes after LoRA.

### FluxEngine.generate() Changes

Extend the signature and generation call:

```python
def generate(
    self,
    prompt: str,
    output_path: Path,
    width: int = 1280,
    height: int = 720,
    seed: Optional[int] = None,
    lora_path: Optional[str] = None,
    lora_weight: float = 0.8,
    # --- NEW ---
    ip_adapter_images: Optional[list[Path]] = None,
    ip_adapter_scales: Optional[list[float]] = None,
    ip_adapter_embeddings: Optional[list[Path]] = None,
) -> ImageResult:
    """
    Generate a single image and save it as PNG.

    IP-Adapter args (FLUX only, ignored if IP-Adapter not loaded):
        ip_adapter_images: Reference image paths for IP-Adapter conditioning.
        ip_adapter_scales: Per-image influence scale (0.0–1.0).
        ip_adapter_embeddings: Pre-computed embedding paths (.ipadpt files).
            If provided, these are used instead of ip_adapter_images (faster).
    """
    if self._pipe is None:
        raise RuntimeError("Call load() before generate()")

    # ... existing LoRA setup code ...

    # --- NEW: Prepare IP-Adapter conditioning ---
    ipa_kwargs = {}
    if getattr(self, "_ip_adapter_loaded", False):
        ipa_kwargs = self._prepare_ip_adapter(
            ip_adapter_images, ip_adapter_scales, ip_adapter_embeddings,
        )

    result = self._pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=self.config.effective_steps,
        guidance_scale=self.config.guidance_scale,
        generator=generator,
        **ipa_kwargs,    # NEW — ip_adapter_image + ip_adapter_image_embeds
    ).images[0]

    # ... existing save + cleanup code ...
```

### IP-Adapter Preparation Helper

```python
def _prepare_ip_adapter(
    self,
    images: Optional[list[Path]],
    scales: Optional[list[float]],
    embeddings: Optional[list[Path]],
) -> dict:
    """Build kwargs for the pipeline call.

    If pre-computed embeddings are provided, use them directly (skips the
    image encoder entirely — faster for repeated generations with the same
    reference).

    Returns a dict with ip_adapter_image or ip_adapter_image_embeds plus
    the scale setting.
    """
    import torch

    kwargs = {}

    if embeddings:
        # Load pre-computed embeddings
        loaded = []
        for emb_path in embeddings:
            loaded.append(torch.load(emb_path, map_location="cpu", weights_only=True))
        kwargs["ip_adapter_image_embeds"] = loaded

    elif images:
        from PIL import Image
        loaded_images = []
        for img_path in images:
            loaded_images.append(Image.open(img_path).convert("RGB"))
        kwargs["ip_adapter_image"] = loaded_images if len(loaded_images) > 1 else loaded_images[0]

    else:
        return {}  # No IP-Adapter conditioning for this generation

    # Set scale(s)
    if scales:
        # For single adapter loaded once, set_ip_adapter_scale accepts a float or list
        if len(scales) == 1:
            self._pipe.set_ip_adapter_scale(scales[0])
        else:
            self._pipe.set_ip_adapter_scale(scales)
    else:
        self._pipe.set_ip_adapter_scale(self.config.ip_adapter.default_scale)

    return kwargs
```

### FluxEngine.unload() Changes

```python
def unload(self) -> None:
    if self._pipe is not None:
        if self._loaded_lora is not None:
            self._pipe.unload_lora_weights()
        # IP-Adapter is unloaded with the pipeline — no separate step needed
        del self._pipe
        self._pipe = None
    self._loaded_lora = None
    self._ip_adapter_loaded = False   # NEW
    clear_vram()
    log.info("FLUX engine unloaded")
```

### ImageEngine Base Class

Add the new params to the `ImageEngine` abstract base class with defaults so `ZImageEngine` doesn't need changes:

```python
# imaging/base.py
class ImageEngine(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        output_path: Path,
        width: int = 1280,
        height: int = 720,
        seed: Optional[int] = None,
        lora_path: Optional[str] = None,
        lora_weight: float = 0.8,
        ip_adapter_images: Optional[list[Path]] = None,
        ip_adapter_scales: Optional[list[float]] = None,
        ip_adapter_embeddings: Optional[list[Path]] = None,
    ) -> ImageResult:
        ...
```

`ZImageEngine.generate()` receives these params but ignores them (IP-Adapter not supported on Z-Image). Log a warning if they're passed:

```python
if ip_adapter_images or ip_adapter_embeddings:
    log.warning("IP-Adapter not supported on Z-Image engine — ignoring IP-Adapter conditioning")
```

### Tests — Phase 3

```python
# tests/test_ip_adapter.py

class TestFluxIPAdapter:
    @patch("musicvision.imaging.flux_engine.clear_vram")
    def test_generate_with_ip_adapter_images(self, mock_clear, tmp_path, gpu_device_map):
        cfg = ImageGenConfig(ip_adapter=IPAdapterConfig(enabled=True))
        engine = FluxEngine(cfg, gpu_device_map)
        mock_pipe, mock_image = _mock_pipe()
        engine._pipe = mock_pipe
        engine._ip_adapter_loaded = True

        # Create fake reference image
        ref_img = tmp_path / "ref.png"
        _create_test_image(ref_img)  # helper that creates a small PIL image

        output = tmp_path / "out.png"
        engine.generate(
            "test prompt", output_path=output,
            ip_adapter_images=[ref_img],
            ip_adapter_scales=[0.7],
        )

        call_kwargs = mock_pipe.call_args[1]
        assert "ip_adapter_image" in call_kwargs
        mock_pipe.set_ip_adapter_scale.assert_called_once_with(0.7)

    def test_generate_without_ip_adapter_no_kwargs(self, tmp_path, gpu_device_map):
        """When no IP-Adapter images provided, no IPA kwargs passed."""
        cfg = ImageGenConfig()
        engine = FluxEngine(cfg, gpu_device_map)
        mock_pipe, mock_image = _mock_pipe()
        engine._pipe = mock_pipe

        output = tmp_path / "out.png"
        engine.generate("test", output_path=output)

        call_kwargs = mock_pipe.call_args[1]
        assert "ip_adapter_image" not in call_kwargs
        assert "ip_adapter_image_embeds" not in call_kwargs

    def test_zimage_ignores_ip_adapter(self, tmp_path, gpu_device_map):
        """Z-Image logs warning but doesn't crash when IPA params passed."""
        cfg = ImageGenConfig(model=ImageModel.ZIMAGE)
        engine = ZImageEngine(cfg, gpu_device_map)
        mock_pipe, mock_image = _mock_pipe()
        engine._pipe = mock_pipe

        ref_img = tmp_path / "ref.png"
        _create_test_image(ref_img)

        output = tmp_path / "out.png"
        # Should not raise
        engine.generate("test", output_path=output, ip_adapter_images=[ref_img])
```

---

## Phase 4 — Consistency Resolver

Create `src/musicvision/assets/consistency.py`:

```python
"""
Automatic consistency resolution for scene image generation.

Given a scene's referenced assets, resolves what conditioning to apply:
- LoRA: first character with a LoRA (only one at a time for FLUX)
- IP-Adapter: all assets with IP-Adapter enabled (multi-IPA supported on FLUX)
- Prompt fragments: always injected regardless of conditioning method
- Engine awareness: Z-Image gets LoRA only; FLUX gets LoRA + IP-Adapter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from musicvision.models import AssetDef, ConsistencyMethod, ImageModel, Scene, StyleSheet


@dataclass
class SceneConditioning:
    """Everything needed to condition a single image generation call."""

    # Prompt fragments from asset descriptions — always used
    prompt_fragments: list[str] = field(default_factory=list)

    # LoRA — at most one active (first character with LoRA wins)
    lora_path: Optional[str] = None
    lora_weight: float = 0.8

    # IP-Adapter — can have multiple (one per asset)
    ip_adapter_images: list[Path] = field(default_factory=list)
    ip_adapter_scales: list[float] = field(default_factory=list)
    ip_adapter_embeddings: list[Path] = field(default_factory=list)

    @property
    def has_lora(self) -> bool:
        return self.lora_path is not None

    @property
    def has_ip_adapter(self) -> bool:
        return bool(self.ip_adapter_images) or bool(self.ip_adapter_embeddings)


def resolve_scene_conditioning(
    scene: Scene,
    style_sheet: StyleSheet,
    image_engine: ImageModel,
    project_root: Optional[Path] = None,
) -> SceneConditioning:
    """
    Build conditioning for a scene from its referenced assets.

    Rules:
    - LoRA: first character/prop/location with has_lora wins. Only one LoRA
      can be active at a time on FLUX (fuse/unfuse). Characters checked first.
    - IP-Adapter: ALL assets with has_ip_adapter contribute reference images.
      Multi-IPA is supported on FLUX. Skipped entirely for Z-Image engines.
    - Prompt fragments: ALL asset descriptions are always injected.
    - Style LoRA (project-level) is handled separately by FluxEngine — NOT
      included here. This function only handles per-scene, per-asset conditioning.

    Args:
        scene: The scene being generated.
        style_sheet: Project style sheet with asset definitions.
        image_engine: Which image model is active (determines IPA support).
        project_root: Project root for resolving relative paths.
    """
    cond = SceneConditioning()

    ipa_supported = image_engine in (ImageModel.FLUX_DEV, ImageModel.FLUX_SCHNELL)

    # Collect all asset IDs referenced by this scene, characters first
    all_ids = list(scene.characters) + list(scene.props) + list(scene.settings)

    for asset_id in all_ids:
        asset = style_sheet.get_asset(asset_id)
        if asset is None:
            continue

        # Always inject description into prompt
        if asset.description:
            cond.prompt_fragments.append(asset.description)

        # LoRA — first one wins
        if asset.has_lora and cond.lora_path is None:
            cond.lora_path = asset.lora_path
            cond.lora_weight = asset.lora_weight

        # IP-Adapter — collect all
        if asset.has_ip_adapter and ipa_supported:
            if asset.ip_adapter_embedding_path:
                emb_path = Path(asset.ip_adapter_embedding_path)
                if project_root and not emb_path.is_absolute():
                    emb_path = project_root / emb_path
                if emb_path.exists():
                    cond.ip_adapter_embeddings.append(emb_path)
                    cond.ip_adapter_scales.append(asset.ip_adapter_scale)
                    continue  # prefer embedding over raw image

            primary = asset.primary_image
            if primary:
                img_path = Path(primary.filename)
                if project_root and not img_path.is_absolute():
                    img_path = project_root / img_path
                if img_path.exists():
                    cond.ip_adapter_images.append(img_path)
                    cond.ip_adapter_scales.append(asset.ip_adapter_scale)

    return cond
```

### Tests — Phase 4

```python
# tests/test_consistency.py

class TestResolveSceneConditioning:
    def _make_scene(self, characters=None, props=None, settings=None):
        return Scene(
            id="s1", order=1, time_start=0, time_end=3.0,
            characters=characters or [], props=props or [], settings=settings or [],
        )

    def _make_stylesheet(self, assets):
        return StyleSheet(assets=assets)

    def test_prompt_only(self):
        ss = self._make_stylesheet([
            AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER,
                     description="Woman with red hair"),
        ])
        scene = self._make_scene(characters=["singer"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV)

        assert cond.prompt_fragments == ["Woman with red hair"]
        assert not cond.has_lora
        assert not cond.has_ip_adapter

    def test_lora_selected(self):
        ss = self._make_stylesheet([
            AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER,
                     description="Singer", consistency=ConsistencyMethod.LORA,
                     lora_path="loras/singer.safetensors", lora_weight=0.9),
        ])
        scene = self._make_scene(characters=["singer"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV)

        assert cond.lora_path == "loras/singer.safetensors"
        assert cond.lora_weight == 0.9

    def test_first_lora_wins(self):
        ss = self._make_stylesheet([
            AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER,
                     description="Singer", consistency=ConsistencyMethod.LORA,
                     lora_path="loras/singer.safetensors"),
            AssetDef(id="dancer", name="Dancer", asset_type=AssetType.CHARACTER,
                     description="Dancer", consistency=ConsistencyMethod.LORA,
                     lora_path="loras/dancer.safetensors"),
        ])
        scene = self._make_scene(characters=["singer", "dancer"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV)

        assert cond.lora_path == "loras/singer.safetensors"

    def test_ip_adapter_collected(self, tmp_path):
        ref_img = tmp_path / "ref.png"
        ref_img.write_bytes(b"fake")
        rel = str(ref_img)

        ss = self._make_stylesheet([
            AssetDef(id="sidekick", name="Sidekick", asset_type=AssetType.CHARACTER,
                     description="Sidekick", consistency=ConsistencyMethod.IP_ADAPTER,
                     ip_adapter_scale=0.5,
                     images=[AssetImage(filename=rel, is_primary=True)]),
        ])
        scene = self._make_scene(characters=["sidekick"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV)

        assert len(cond.ip_adapter_images) == 1
        assert cond.ip_adapter_scales == [0.5]

    def test_ip_adapter_skipped_for_zimage(self, tmp_path):
        ref_img = tmp_path / "ref.png"
        ref_img.write_bytes(b"fake")

        ss = self._make_stylesheet([
            AssetDef(id="sidekick", name="Sidekick", asset_type=AssetType.CHARACTER,
                     consistency=ConsistencyMethod.IP_ADAPTER,
                     images=[AssetImage(filename=str(ref_img), is_primary=True)]),
        ])
        scene = self._make_scene(characters=["sidekick"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.ZIMAGE)

        assert not cond.has_ip_adapter  # Z-Image doesn't support IPA

    def test_both_lora_and_ip_adapter(self, tmp_path):
        ref_img = tmp_path / "ref.png"
        ref_img.write_bytes(b"fake")

        ss = self._make_stylesheet([
            AssetDef(id="hero", name="Hero", asset_type=AssetType.CHARACTER,
                     consistency=ConsistencyMethod.BOTH,
                     lora_path="loras/hero.safetensors",
                     ip_adapter_scale=0.4,
                     images=[AssetImage(filename=str(ref_img), is_primary=True)]),
        ])
        scene = self._make_scene(characters=["hero"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV)

        assert cond.has_lora
        assert cond.has_ip_adapter

    def test_missing_asset_ignored(self):
        ss = self._make_stylesheet([])
        scene = self._make_scene(characters=["nonexistent"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV)

        assert cond.prompt_fragments == []
        assert not cond.has_lora
        assert not cond.has_ip_adapter
```

---

## Phase 5 — Wire Conditioning into Generation

### Replace LoRA-only Lookup

The current image generation code in `cli.py` and `api/app.py` has a manual LoRA lookup loop:

```python
# CURRENT (cli.py and api/app.py):
char_loras: dict[str, tuple[str, float]] = {}
for char_def in svc.config.style_sheet.characters:
    if char_def.lora_path:
        char_loras[char_def.id] = (char_def.lora_path, char_def.lora_weight)
# ... then manually picks first matching LoRA per scene
```

Replace with:

```python
# NEW:
from musicvision.assets.consistency import resolve_scene_conditioning

cond = resolve_scene_conditioning(
    scene, proj.config.style_sheet, proj.config.image_gen.model,
    project_root=proj.paths.root,
)

# Inject asset descriptions into prompt
full_prompt = prompt
if cond.prompt_fragments:
    # Prepend character/prop/location descriptions
    context = " ".join(cond.prompt_fragments)
    full_prompt = f"{context}. {prompt}"

engine.generate(
    prompt=full_prompt,
    width=width,
    height=height,
    output_path=output_path,
    seed=seed,
    lora_path=cond.lora_path,
    lora_weight=cond.lora_weight,
    ip_adapter_images=cond.ip_adapter_images if cond.has_ip_adapter else None,
    ip_adapter_scales=cond.ip_adapter_scales if cond.has_ip_adapter else None,
    ip_adapter_embeddings=cond.ip_adapter_embeddings if cond.ip_adapter_embeddings else None,
)
```

This replaces the LoRA lookup in:
1. `cmd_images()` in `cli.py`
2. `generate_images()` in `api/app.py`
3. `regenerate_image()` in `api/app.py`

**Sort optimization:** Currently, scenes are sorted by LoRA path to minimize LoRA swaps. Extend this to also group by IP-Adapter reference — not critical since IP-Adapter doesn't require load/unload cycles, but keeps generation ordering predictable.

### Auto-enable IP-Adapter

If any asset has `consistency` set to `ip_adapter` or `both`, the generation code should auto-enable IP-Adapter on the engine config before loading:

```python
# Before engine.load():
has_ipa_assets = any(
    a.has_ip_adapter for a in proj.config.style_sheet.assets
)
if has_ipa_assets and proj.config.image_gen.model in (ImageModel.FLUX_DEV, ImageModel.FLUX_SCHNELL):
    proj.config.image_gen.ip_adapter.enabled = True
```

---

## Phase 6 — Asset CRUD API Endpoints

Add to `api/app.py`:

```python
# --- Request/Response schemas ---

class CreateAssetRequest(BaseModel):
    id: str
    name: str
    asset_type: AssetType
    description: str = ""

class UpdateAssetRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    consistency: Optional[ConsistencyMethod] = None
    lora_path: Optional[str] = None
    lora_weight: Optional[float] = None
    ip_adapter_scale: Optional[float] = None

class UpdateAssetImageRequest(BaseModel):
    role: Optional[str] = None
    caption: Optional[str] = None
    is_primary: Optional[bool] = None

class GenerateAssetImageRequest(BaseModel):
    prompt: Optional[str] = None     # if None, auto-generate from description
    seed: int = -1
    width: int = 1024
    height: int = 1024

# --- Endpoints ---

@app.get("/api/assets")
async def list_assets(asset_type: Optional[str] = None) -> list[AssetDef]:
    proj = get_project()
    svc = AssetService(proj)
    at = AssetType(asset_type) if asset_type else None
    return svc.list_assets(at)

@app.post("/api/assets")
async def create_asset(req: CreateAssetRequest) -> AssetDef:
    proj = get_project()
    svc = AssetService(proj)
    return svc.create_asset(req.id, req.name, req.asset_type, req.description)

@app.get("/api/assets/{asset_id}")
async def get_asset(asset_id: str) -> AssetDef:
    proj = get_project()
    asset = proj.config.style_sheet.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' not found")
    return asset

@app.put("/api/assets/{asset_id}")
async def update_asset(asset_id: str, req: UpdateAssetRequest) -> AssetDef:
    proj = get_project()
    svc = AssetService(proj)
    updates = req.model_dump(exclude_none=True)
    return svc.update_asset(asset_id, **updates)

@app.delete("/api/assets/{asset_id}")
async def delete_asset(asset_id: str):
    proj = get_project()
    svc = AssetService(proj)
    svc.delete_asset(asset_id)
    return {"status": "deleted", "asset_id": asset_id}

@app.post("/api/assets/{asset_id}/images")
async def upload_asset_image(
    asset_id: str,
    file: UploadFile,
    role: str = "reference",
    caption: str = "",
    is_primary: bool = False,
) -> AssetImage:
    proj = get_project()
    svc = AssetService(proj)

    # Save upload to temp, then add via service
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        return svc.add_image(asset_id, tmp_path, role=role, caption=caption, is_primary=is_primary)
    finally:
        tmp_path.unlink(missing_ok=True)

@app.delete("/api/assets/{asset_id}/images/{filename:path}")
async def remove_asset_image(asset_id: str, filename: str):
    proj = get_project()
    svc = AssetService(proj)
    svc.remove_image(asset_id, filename)
    return {"status": "deleted"}

@app.put("/api/assets/{asset_id}/images/{filename:path}")
async def update_asset_image(asset_id: str, filename: str, req: UpdateAssetImageRequest) -> AssetDef:
    proj = get_project()
    asset = proj.config.style_sheet.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' not found")

    img = next((i for i in asset.images if i.filename == filename), None)
    if not img:
        raise HTTPException(status_code=404, detail=f"Image '{filename}' not found")

    if req.role is not None:
        img.role = req.role
    if req.caption is not None:
        img.caption = req.caption
    if req.is_primary is True:
        svc = AssetService(proj)
        svc.set_primary_image(asset_id, filename)
    proj.save_config()
    return asset

@app.post("/api/assets/{asset_id}/generate-image")
async def generate_asset_image(asset_id: str, req: GenerateAssetImageRequest) -> AssetDef:
    """Generate a reference image for an asset using the current image engine."""
    from musicvision.imaging import create_engine
    from musicvision.utils.gpu import detect_devices

    proj = get_project()
    asset = proj.config.style_sheet.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' not found")

    prompt = req.prompt or asset.description
    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt or description available")

    # Inject style context
    ss = proj.config.style_sheet
    if ss.visual_style:
        prompt = f"{prompt}. {ss.visual_style}"
    if ss.color_palette:
        prompt = f"{prompt}. Color palette: {ss.color_palette}"

    def _run():
        device_map = detect_devices()
        engine = create_engine(proj.config.image_gen, device_map)
        engine.load()
        try:
            output_dir = proj.paths.asset_dir(asset.asset_type, asset_id)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Auto-name
            existing = list(output_dir.glob("generated_*.png"))
            idx = len(existing) + 1
            output_path = output_dir / f"generated_{idx:03d}.png"

            seed = req.seed if req.seed >= 0 else None
            engine.generate(
                prompt=prompt,
                output_path=output_path,
                width=req.width,
                height=req.height,
                seed=seed,
                lora_path=ss.style_lora_path,
                lora_weight=ss.style_lora_weight,
            )

            # Register as reference image
            rel_path = str(output_path.relative_to(proj.paths.root))
            is_primary = not asset.images  # first image becomes primary
            img = AssetImage(filename=rel_path, role="reference", is_primary=is_primary)
            asset.images.append(img)
            proj.save_config()
        finally:
            engine.unload()
        return asset

    import asyncio
    return await asyncio.to_thread(_run)
```

---

## Phase 7 — Embedding Precomputation & Caching

### API Endpoint

```python
@app.post("/api/assets/{asset_id}/precompute-embedding")
async def precompute_embedding(asset_id: str) -> AssetDef:
    """Precompute and cache IP-Adapter embedding for an asset's primary image."""
    from musicvision.imaging import create_engine
    from musicvision.imaging.flux_engine import FluxEngine
    from musicvision.utils.gpu import detect_devices

    proj = get_project()
    asset = proj.config.style_sheet.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' not found")
    if not asset.primary_image:
        raise HTTPException(status_code=400, detail="Asset has no reference images")
    if proj.config.image_gen.model not in (ImageModel.FLUX_DEV, ImageModel.FLUX_SCHNELL):
        raise HTTPException(status_code=400, detail="IP-Adapter embedding requires FLUX engine")

    def _run():
        import torch
        from PIL import Image

        device_map = detect_devices()
        config = proj.config.image_gen.model_copy()
        config.ip_adapter.enabled = True
        engine = FluxEngine(config, device_map)
        engine.load()

        try:
            img_path = proj.resolve_path(asset.primary_image.filename)
            image = Image.open(img_path).convert("RGB")

            # Precompute embedding using diffusers helper
            embeds = engine._pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=image,
                ip_adapter_image_embeds=None,
                device=device_map.primary,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )

            # Save to cache
            cache_dir = proj.paths.ip_cache_dir
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{asset_id}.ipadpt"
            torch.save(embeds, cache_path)

            # Update asset
            asset.ip_adapter_embedding_path = str(cache_path.relative_to(proj.paths.root))
            proj.save_config()

        finally:
            engine.unload()

        return asset

    import asyncio
    return await asyncio.to_thread(_run)
```

### Cache Invalidation

When an asset's primary image changes (upload, delete, set_primary), invalidate the cached embedding. This is already handled by `AssetService.invalidate_embedding_cache()` — call it from:
- `add_image()` when `is_primary=True`
- `remove_image()` when the removed image was primary
- `set_primary_image()`

---

## Phase 8 — React Asset Library Panel

### New Components

```
frontend/src/components/
├── AssetLibrary.tsx         # Main panel — grid of asset cards grouped by type
├── AssetCard.tsx            # Thumbnail card for a single asset
├── AssetDetail.tsx          # Detail/edit view for a single asset
├── AssetImageManager.tsx    # Upload, reorder, set primary, delete images
├── AssetPicker.tsx          # Modal for tagging assets to scenes
└── ConsistencyBadge.tsx     # Visual indicator: None / IPA / LoRA / Both
```

### AssetLibrary Panel

- Three sections: **Characters**, **Props**, **Locations**
- Each section shows a grid of `AssetCard` components
- "Add Asset" button per section opens a create form (id, name, description)
- Click card → expand to `AssetDetail` inline or as slide-over panel

### AssetCard

- Thumbnail: primary image or placeholder icon (person / cube / map-pin)
- Asset name
- `ConsistencyBadge` showing active method
- Quick actions: edit, delete

### AssetDetail

- Editable fields: name, description
- Image gallery: `AssetImageManager` component
- Consistency method selector (dropdown): None / IP-Adapter / LoRA / Both
  - If LoRA selected but no `lora_path`: show "No LoRA weights — train or provide .safetensors"
  - If IP-Adapter selected but no images: show "Upload a reference image first"
  - If IP-Adapter selected and engine is Z-Image: show "IP-Adapter requires FLUX engine"
- LoRA path field (file picker or text input)
- LoRA weight slider (0.0–1.0)
- IP-Adapter scale slider (0.0–1.0)
- "Generate Reference Image" button → calls `/api/assets/{id}/generate-image`
- "Precompute Embedding" button (shown when IP-Adapter is active and FLUX selected) → calls `/api/assets/{id}/precompute-embedding`

### AssetImageManager

- Drag-and-drop upload zone
- Image thumbnails with:
  - Star icon for primary (click to set)
  - Role badge (reference / training)
  - Caption field (editable inline, saved on blur)
  - Delete button
- "Upload Images" button (multi-file)

### Scene ↔ Asset Linking

The existing scene editor already has `characters`, `props`, `settings` fields. Update the scene edit form to show an `AssetPicker`:

- Shows available assets grouped by type
- Checkboxes for selecting which assets are in the scene
- Selected assets appear as chips/tags on the scene row in the storyboard
- When a scene is selected for image generation, the storyboard shows what conditioning will be applied (LoRA badge, IPA badge, or just prompt text)

### API Client Updates

Add to `frontend/src/api/client.ts`:

```typescript
// Asset CRUD
export const listAssets = (type?: AssetType) =>
  fetch(`/api/assets${type ? `?asset_type=${type}` : ""}`).then(r => r.json());

export const createAsset = (data: CreateAssetRequest) =>
  fetch("/api/assets", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(data) })
    .then(r => r.json());

export const getAsset = (id: string) =>
  fetch(`/api/assets/${id}`).then(r => r.json());

export const updateAsset = (id: string, data: UpdateAssetRequest) =>
  fetch(`/api/assets/${id}`, { method: "PUT", headers: JSON_HEADERS, body: JSON.stringify(data) })
    .then(r => r.json());

export const deleteAsset = (id: string) =>
  fetch(`/api/assets/${id}`, { method: "DELETE" }).then(r => r.json());

// Asset images
export const uploadAssetImage = (id: string, file: File, role = "reference", caption = "", isPrimary = false) => {
  const form = new FormData();
  form.append("file", file);
  form.append("role", role);
  form.append("caption", caption);
  form.append("is_primary", String(isPrimary));
  return fetch(`/api/assets/${id}/images`, { method: "POST", body: form }).then(r => r.json());
};

export const deleteAssetImage = (id: string, filename: string) =>
  fetch(`/api/assets/${id}/images/${encodeURIComponent(filename)}`, { method: "DELETE" })
    .then(r => r.json());

export const generateAssetImage = (id: string, data?: GenerateAssetImageRequest) =>
  fetch(`/api/assets/${id}/generate-image`, { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(data || {}) })
    .then(r => r.json());

export const precomputeEmbedding = (id: string) =>
  fetch(`/api/assets/${id}/precompute-embedding`, { method: "POST" }).then(r => r.json());
```

---

## What NOT to Do

- **No LoRA training in this spec.** The asset library stores training datasets, but training execution is a separate panel and spec.
- **No ControlNet / pose conditioning.** That's a future consistency layer (see `future_plans.md`).
- **No Z-Image IP-Adapter hacks.** The SD3 adapter workaround is experimental and unsupported. If Z-Image gets official IP-Adapter support, it can be added to `ZImageEngine` later.
- **No multi-LoRA stacking.** FLUX supports loading multiple LoRAs but fusing/unfusing is complex. One character LoRA + one style LoRA (project-level) is the current scope.
- **No automatic consistency method selection.** The user explicitly sets `consistency` per asset. The system doesn't auto-promote from "none" to "ip_adapter" just because images exist.
- **No image generation from within `AssetService`.** Image generation goes through the API endpoint → engine, not through the service layer. AssetService is pure data management.

---

## Dependency Changes

Add to `pyproject.toml` (if not already present):

```toml
# No new dependencies required.
# IP-Adapter support comes from diffusers (already installed).
# CLIP vision model downloaded on first use via transformers (already installed).
# torch.save/load for embedding caching (already installed).
```

The XLabs IP-Adapter weights (~982 MB) are downloaded on first use from HuggingFace. The CLIP-ViT-L encoder (~600 MB) is downloaded similarly. Both are cached by the HuggingFace hub.

---

## IP-Adapter Model Selection

The spec defaults to XLabs v1 (`XLabs-AI/flux-ip-adapter`). Alternatives to evaluate:

| Adapter | Encoder | Training | License | Notes |
|---------|---------|----------|---------|-------|
| XLabs v1 | CLIP-ViT-L | 50k+25k steps | FLUX-dev NC | Mature, well-tested in ComfyUI |
| XLabs v2 | CLIP-ViT-L | 150k+350k steps | FLUX-dev NC | More training, may be better quality |
| InstantX | SigLIP-so400m | Unknown | FLUX-dev NC | Better face fidelity reported |

The `IPAdapterConfig.model_repo` and `image_encoder_repo` fields make it easy to swap. Test all three on your actual storyboard scenes before committing.

---

## Migration Checklist for Existing Projects

1. Open existing project → `StyleSheet` validator auto-migrates `characters`/`props`/`settings` to `assets`
2. All assets start with `consistency: none` (existing behavior preserved)
3. User can then optionally upgrade assets: add reference images, set consistency to `ip_adapter` or `lora`
4. No breaking changes — existing `Scene.characters`, `.props`, `.settings` fields still work
5. Old `CharacterDef.lora_path` is migrated to `AssetDef.lora_path` with `consistency: lora`
