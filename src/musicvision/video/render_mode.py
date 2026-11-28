"""Render mode → per-engine config for one generation run.

``render_mode`` is the job-level speed/quality switch ("preview" or "final")
that the API and UI expose. It never mutates the project config: callers get
a fresh copy to hand to the engine, so nothing ephemeral can leak into
project.yaml on a later save.

- LTX-Video 2: preview = 480x320 / 20 steps, final = 768x512 / 40 steps.
- HuMo: preview = the HumoQuality.PREVIEW fast-iteration settings (480p,
  6 steps, Lightx2V distill LoRA, CFG off) on the project's own tier;
  final = the project's HuMo settings unchanged.
"""

from __future__ import annotations

from typing import Literal

from musicvision.models import HumoConfig, HumoQuality, LtxVideoConfig, ProjectConfig, VideoEngineType

RenderMode = Literal["preview", "final"]

# Fields taken from the HuMo preview preset. ``tier`` is deliberately excluded:
# the project's tier reflects what fits in VRAM and what weights are present.
_HUMO_PREVIEW_FIELDS = ("resolution", "denoising_steps", "shift", "scale_t", "scale_a", "lora")


def config_for_render_mode(
    engine_type: VideoEngineType,
    project_config: ProjectConfig,
    render_mode: RenderMode,
) -> HumoConfig | LtxVideoConfig:
    """Return a copy of the engine config with *render_mode* applied."""
    if render_mode not in ("preview", "final"):
        raise ValueError(f"render_mode must be 'preview' or 'final', got {render_mode!r}")

    if engine_type == VideoEngineType.LTX_VIDEO:
        config = project_config.ltx_video.model_copy(deep=True)
        if render_mode == "preview":
            config.width, config.height, config.num_inference_steps = 480, 320, 20
        else:
            config.width, config.height, config.num_inference_steps = 768, 512, 40
        return config

    config = project_config.humo.model_copy(deep=True)
    if render_mode == "preview":
        preset = HumoConfig.from_quality(HumoQuality.PREVIEW)
        for field in _HUMO_PREVIEW_FIELDS:
            setattr(config, field, getattr(preset, field))
    return config


def missing_render_weights(config: HumoConfig | LtxVideoConfig) -> dict | None:
    """Describe weights the config needs that are not on disk, or None.

    The HuMo loader only warns and continues when a LoRA is missing; for the
    preview preset that would run 6 undistilled steps and produce noise, so
    callers should fail the job instead.
    """
    lora = getattr(config, "lora", None)
    if not lora:
        return None
    from musicvision.video.weight_registry import locate_lora

    try:
        locate_lora(lora)
    except (FileNotFoundError, KeyError):
        return {
            "missing": "lora",
            "lora": lora,
            "hint": (
                f"Download it with musicvision.video.weight_registry.download_lora({lora!r}), "
                "or use render_mode='final'."
            ),
        }
    return None
