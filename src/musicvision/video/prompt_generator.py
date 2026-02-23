"""
LLM-assisted video prompt generation for HuMo.

Generates dense, descriptive prompts matching HuMo's expected style
(similar to Qwen2.5-VL training captions).

From HUMO_REFERENCE.md:
  DO: appearance detail, environment, lighting, action/motion, camera framing
  DON'T: abstract language, temporal instructions, audio descriptions

Backend controlled by LLM_BACKEND env var. See musicvision/llm.py for config.
"""

from __future__ import annotations

import logging

from musicvision.llm import LLMClient, LLMConfig, get_client
from musicvision.models import ProjectConfig, Scene, SceneType

log = logging.getLogger(__name__)

VIDEO_PROMPT_SYSTEM = """You are a music video director writing video generation prompts for HuMo.

HuMo is a video diffusion model trained on dense, factual visual descriptions
in the style of Qwen2.5-VL automatic captions.

DO include:
- Subject/character appearance (clothing, hair, expression, body language)
- Precise environment and background detail
- Lighting conditions (quality, direction, colour)
- Motion and action (what is physically happening, camera movement)
- Camera framing (close-up, wide shot, angle, depth of field)

DO NOT include:
- Song or audio references ("as the music swells...")
- Temporal instructions ("then", "next", "suddenly")
- Abstract emotional language ("hopeful", "melancholic")
- Instructions to the model ("generate", "create", "render")

Length: 3–5 dense sentences, 80–160 words.
Output only the prompt text itself — no commentary, headers, or markdown."""


def _build_style_context(config: ProjectConfig) -> str:
    """Serialize the style sheet into a compact text block for the LLM."""
    ss = config.style_sheet
    parts: list[str] = []

    if ss.visual_style:
        parts.append(f"Visual style: {ss.visual_style}")
    if ss.color_palette:
        parts.append(f"Color palette: {ss.color_palette}")
    if ss.characters:
        for c in ss.characters:
            parts.append(f"Character — {c.id}: {c.description}")
    if ss.props:
        for p in ss.props:
            parts.append(f"Prop — {p.id}: {p.description}")
    if ss.settings:
        for s in ss.settings:
            parts.append(f"Setting — {s.id}: {s.description}")

    return "\n".join(parts) if parts else "(no style sheet defined)"


def generate_video_prompt(
    scene: Scene,
    config: ProjectConfig,
    context_scenes: list[Scene] | None = None,
    llm_config: LLMConfig | None = None,
) -> str:
    """
    Generate a HuMo video conditioning prompt for a scene.

    More detailed than the image prompt — specifies motion, camera movement,
    expressions, and physical action using the dense descriptive style HuMo
    was trained on (Qwen2.5-VL captions).

    Args:
        scene: The scene to generate a prompt for
        config: Project config containing the style sheet
        context_scenes: Adjacent scenes for visual continuity (optional)
        llm_config: Explicit LLM config; falls back to env vars if None

    Returns:
        HuMo-compatible video prompt string
    """
    client: LLMClient = get_client(llm_config)

    style_context = _build_style_context(config)

    scene_type_label = "instrumental (no lyrics)" if scene.type == SceneType.INSTRUMENTAL else "vocal"
    duration_note = (
        f"{scene.duration:.1f}s"
        + (" — will be split into sub-clips" if scene.needs_sub_clips else "")
    )

    # Include the approved reference image prompt as visual anchor if available
    image_prompt_note = ""
    if scene.effective_image_prompt:
        image_prompt_note = (
            f"\n\nReference image prompt (the still this clip starts from):\n"
            f"{scene.effective_image_prompt}"
        )

    user_msg = f"""Style sheet:
{style_context}

Scene to animate:
  ID: {scene.id}
  Type: {scene_type_label}
  Duration: {duration_note}
  Lyrics: {scene.lyrics or '(none)'}
  Section: {scene.notes or '(unspecified)'}"""

    user_msg += image_prompt_note

    if context_scenes:
        context_lines = []
        for cs in context_scenes:
            label = cs.lyrics or "(instrumental)"
            context_lines.append(f"  [{cs.id}] {label}")
        user_msg += "\n\nSurrounding scenes for visual continuity:\n" + "\n".join(context_lines)

    if scene.characters:
        user_msg += f"\n\nCharacters present: {', '.join(scene.characters)}"
    if scene.props:
        user_msg += f"\nProps: {', '.join(scene.props)}"
    if scene.settings:
        user_msg += f"\nSettings: {', '.join(scene.settings)}"

    user_msg += "\n\nWrite the HuMo video prompt for this scene:"

    log.info("Generating video prompt for %s...", scene.id)
    return client.chat(VIDEO_PROMPT_SYSTEM, user_msg)
