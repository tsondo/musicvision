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

from musicvision.models import ImageModel, Scene, StyleSheet

# Engines that support IP-Adapter conditioning (stock diffusers FLUX path).
_IPA_ENGINES = (ImageModel.FLUX_DEV, ImageModel.FLUX_SCHNELL)


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

    def apply_to_prompt(self, prompt: str) -> str:
        """Prepend asset description fragments to a scene prompt."""
        if not self.prompt_fragments:
            return prompt
        context = " ".join(self.prompt_fragments)
        return f"{context}. {prompt}"


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

    ipa_supported = image_engine in _IPA_ENGINES

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


def conditioning_sort_key(cond: SceneConditioning) -> tuple:
    """Stable grouping key for generation ordering.

    Scenes sharing a LoRA generate consecutively (minimizes fuse/unfuse swaps);
    within a LoRA group, scenes sharing the same IP-Adapter reference set are
    grouped too — IPA needs no load/unload cycles, but predictable ordering
    keeps logs and seed sweeps sane.
    """
    return (
        cond.lora_path or "",
        tuple(str(p) for p in cond.ip_adapter_images),
        tuple(str(p) for p in cond.ip_adapter_embeddings),
    )


def should_enable_ip_adapter(style_sheet: StyleSheet, image_engine: ImageModel) -> bool:
    """True if any asset wants IP-Adapter conditioning and the engine supports it.

    Call before engine.load() so the adapter loads at the correct point in the
    load sequence (before any cpu-offload — see FluxEngine load-order audit).
    """
    if image_engine not in _IPA_ENGINES:
        return False
    return any(a.has_ip_adapter for a in style_sheet.assets)
