"""
HuMo weight registry: maps every HumoTier to its HuggingFace sources.

Handles download (huggingface_hub) and local path resolution so every
other module can just call locate_weights(tier) and get back a Path.

Weight layout on disk (all relative to weights_dir, default ~/.cache/musicvision/weights):

  humo/
    fp16/           bytedance-research/HuMo — full-precision safetensors shards
    fp8_scaled/     Kijai/WanVideo_comfy_fp8_scaled/HuMo — single fp8 file
    gguf/           Alissonerdx/Wan2.1-HuMo-GGUF — Q8_0 / Q6_K / Q4_K_M files
    preview/        bytedance-research/HuMo — 1.7B safetensors
  shared/
    t5/             Wan-AI/Wan2.1-T2V-1.3B — UMT5-XXL encoder
    vae/            Wan-AI/Wan2.1-T2V-1.3B — video VAE
    whisper/        openai/whisper-large-v3 — encoder only
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from musicvision.models import HumoTier

log = logging.getLogger(__name__)

# Default cache root.  Override via MUSICVISION_WEIGHTS_DIR env var.
_DEFAULT_WEIGHTS_DIR = Path.home() / ".cache" / "musicvision" / "weights"


@dataclass(frozen=True)
class WeightSpec:
    """Location of a single weight artifact on HuggingFace."""
    repo_id: str
    filename: str                                   # file path within the repo
    fmt: Literal["safetensors", "gguf", "pth"]
    expected_gb: float                              # rough download size for progress
    subfolder: str = ""                             # HF subfolder, if any
    local_subdir: str = ""                          # subdir under weights_dir/{category}/


# ---------------------------------------------------------------------------
# DiT weight specs — one per tier
# ---------------------------------------------------------------------------

_DIT_SPECS: dict[HumoTier, WeightSpec] = {
    HumoTier.FP16: WeightSpec(
        repo_id="bytedance-research/HuMo",
        filename="diffusion_models/",               # directory of shards
        fmt="safetensors",
        expected_gb=34.0,
        local_subdir="fp16",
    ),
    HumoTier.FP8_SCALED: WeightSpec(
        repo_id="Kijai/WanVideo_comfy_fp8_scaled",
        filename="HuMo_14B_fp8_e4m3fn_scaled.safetensors",
        fmt="safetensors",
        expected_gb=18.0,
        subfolder="HuMo",
        local_subdir="fp8_scaled",
    ),
    HumoTier.GGUF_Q8: WeightSpec(
        repo_id="Alissonerdx/Wan2.1-HuMo-GGUF",
        filename="HuMo-Q8_0.gguf",
        fmt="gguf",
        expected_gb=18.5,
        local_subdir="gguf",
    ),
    HumoTier.GGUF_Q6: WeightSpec(
        repo_id="Alissonerdx/Wan2.1-HuMo-GGUF",
        filename="HuMo-Q6_K.gguf",
        fmt="gguf",
        expected_gb=14.4,
        local_subdir="gguf",
    ),
    HumoTier.GGUF_Q4: WeightSpec(
        repo_id="Alissonerdx/Wan2.1-HuMo-GGUF",
        filename="HuMo-Q4_K_M.gguf",
        fmt="gguf",
        expected_gb=11.5,
        local_subdir="gguf",
    ),
    HumoTier.PREVIEW: WeightSpec(
        repo_id="bytedance-research/HuMo",
        filename="diffusion_models_1_3B/",          # 1.7B shard directory
        fmt="safetensors",
        expected_gb=3.4,
        local_subdir="preview",
    ),
}

# ---------------------------------------------------------------------------
# Shared weight specs — tier-independent
# ---------------------------------------------------------------------------

SHARED_SPECS: dict[str, WeightSpec] = {
    # UMT5-XXL text encoder (FP16 ~10 GB, BF16 acceptable)
    "t5": WeightSpec(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        filename="models_t5_umt5-xxl-enc-bf16.pth",
        fmt="pth",
        expected_gb=10.0,
        local_subdir="t5",
    ),
    # Video VAE
    "vae": WeightSpec(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        filename="Wan2.1_VAE.pth",
        fmt="pth",
        expected_gb=0.4,
        local_subdir="vae",
    ),
    # Whisper large-v3 encoder (we only need the encoder)
    "whisper": WeightSpec(
        repo_id="openai/whisper-large-v3",
        filename="model.safetensors",
        fmt="safetensors",
        expected_gb=1.5,
        local_subdir="whisper",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def weights_dir() -> Path:
    """Return root weights directory, honouring MUSICVISION_WEIGHTS_DIR env var."""
    d = Path(os.environ.get("MUSICVISION_WEIGHTS_DIR", _DEFAULT_WEIGHTS_DIR))
    return d


def dit_spec(tier: HumoTier) -> WeightSpec:
    return _DIT_SPECS[tier]


def locate_dit(tier: HumoTier, base_dir: Path | None = None) -> Path:
    """
    Return local path to the DiT weight file/directory for *tier*.

    Raises FileNotFoundError if not present — call download_dit() first.
    """
    base = base_dir or weights_dir()
    spec = _DIT_SPECS[tier]
    p = base / "humo" / spec.local_subdir / spec.filename
    if not p.exists():
        # For directory specs (fp16 / preview), the path is the directory itself
        p_dir = base / "humo" / spec.local_subdir
        if p_dir.is_dir() and any(p_dir.iterdir()):
            return p_dir
        raise FileNotFoundError(
            f"HuMo weights for tier '{tier.value}' not found at {p}. "
            f"Run: musicvision download-weights --tier {tier.value}"
        )
    return p


def locate_shared(key: str, base_dir: Path | None = None) -> Path:
    """Return local path to a shared weight file (t5 / vae / whisper)."""
    base = base_dir or weights_dir()
    spec = SHARED_SPECS[key]
    p = base / "shared" / spec.local_subdir / spec.filename
    if not p.exists():
        raise FileNotFoundError(
            f"Shared weight '{key}' not found at {p}. "
            f"Run: musicvision download-weights --tier <any>"
        )
    return p


def download_dit(
    tier: HumoTier,
    base_dir: Path | None = None,
    hf_token: str | None = None,
) -> Path:
    """
    Download DiT weights for *tier* from HuggingFace if not already present.

    Returns the local path.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is not installed. Run: pip install huggingface-hub"
        ) from e

    base = base_dir or weights_dir()
    spec = _DIT_SPECS[tier]
    local_dir = base / "humo" / spec.local_subdir
    local_dir.mkdir(parents=True, exist_ok=True)

    token = hf_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    if spec.filename.endswith("/"):
        # Directory of shards — use snapshot_download with allow_patterns
        log.info(
            "Downloading HuMo %s weights from %s (~%.0f GB)…",
            tier.value, spec.repo_id, spec.expected_gb,
        )
        snapshot_download(
            repo_id=spec.repo_id,
            allow_patterns=[f"{spec.filename}*"],
            local_dir=str(local_dir),
            token=token,
        )
        return local_dir
    else:
        dest = local_dir / spec.filename
        if dest.exists():
            log.info("HuMo %s weights already cached at %s", tier.value, dest)
            return dest
        log.info(
            "Downloading HuMo %s weights from %s (~%.0f GB)…",
            tier.value, spec.repo_id, spec.expected_gb,
        )
        hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            subfolder=spec.subfolder or None,
            local_dir=str(local_dir),
            token=token,
        )
        return dest


def download_shared(
    key: str,
    base_dir: Path | None = None,
    hf_token: str | None = None,
) -> Path:
    """Download a shared weight file if not already present."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError("huggingface_hub is not installed. Run: pip install huggingface-hub") from e

    base = base_dir or weights_dir()
    spec = SHARED_SPECS[key]
    local_dir = base / "shared" / spec.local_subdir
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / spec.filename

    if dest.exists():
        log.info("Shared weight '%s' already cached at %s", key, dest)
        return dest

    token = hf_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    log.info("Downloading shared weight '%s' from %s (~%.1f GB)…", key, spec.repo_id, spec.expected_gb)
    hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.filename,
        local_dir=str(local_dir),
        token=token,
    )
    return dest


def download_all_for_tier(
    tier: HumoTier,
    base_dir: Path | None = None,
    hf_token: str | None = None,
) -> dict[str, Path]:
    """Download DiT + all shared weights for *tier*. Returns {key: path}."""
    paths: dict[str, Path] = {}
    paths["dit"] = download_dit(tier, base_dir=base_dir, hf_token=hf_token)
    for key in SHARED_SPECS:
        paths[key] = download_shared(key, base_dir=base_dir, hf_token=hf_token)
    return paths


def weight_status(tier: HumoTier, base_dir: Path | None = None) -> dict[str, bool]:
    """Return presence status for all weights required by *tier*."""
    status: dict[str, bool] = {}
    try:
        locate_dit(tier, base_dir)
        status["dit"] = True
    except FileNotFoundError:
        status["dit"] = False
    for key in SHARED_SPECS:
        try:
            locate_shared(key, base_dir)
            status[key] = True
        except FileNotFoundError:
            status[key] = False
    return status
