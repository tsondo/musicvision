"""
LLM-assisted image prompt generation.

Generates FLUX prompts for each scene, injecting style sheet elements
(characters, props, settings) for visual consistency.

Backend controlled by LLM_BACKEND env var. See musicvision/llm.py for config.
"""

from __future__ import annotations

import logging

from musicvision.llm import LLMClient, LLMConfig, get_client
from musicvision.models import ProjectConfig, Scene, SceneType

log = logging.getLogger(__name__)

IMAGE_PROMPT_SYSTEM = """You are a music video art director writing image generation prompts for FLUX.

Each prompt describes a single cinematic still frame for one scene of a music video.

Guidelines:
- Be concrete and visual — describe what the camera literally sees
- Include: subject/subject appearance, environment, lighting quality, mood, composition/framing
- Length: 2–4 sentences, 60–130 words
- Match the emotional tone of the scene's lyrics
- If the scene is instrumental, focus on atmosphere and setting
- Avoid: abstract concepts, song/audio references, temporal language ("then", "next")
- Do NOT include any intro, commentary, or markdown — output only the prompt text itself"""


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


def generate_image_prompt(
    scene: Scene,
    config: ProjectConfig,
    context_scenes: list[Scene] | None = None,
    llm_config: LLMConfig | None = None,
) -> str:
    """
    Generate a FLUX image prompt for a single scene.

    Considers:
      - Scene lyrics and type (vocal/instrumental)
      - Style sheet (visual style, color palette, characters, props, settings)
      - Surrounding scenes for narrative coherence (optional)

    Args:
        scene: The scene to generate a prompt for
        config: Project config containing the style sheet
        context_scenes: Adjacent scenes for narrative context (optional)
        llm_config: Explicit LLM config; falls back to env vars if None

    Returns:
        FLUX-compatible image prompt string
    """
    client: LLMClient = get_client(llm_config)

    style_context = _build_style_context(config)

    scene_type_label = "instrumental (no lyrics)" if scene.type == SceneType.INSTRUMENTAL else "vocal"

    user_msg = f"""Style sheet:
{style_context}

Scene to describe:
  ID: {scene.id}
  Type: {scene_type_label}
  Duration: {scene.duration:.1f}s
  Lyrics: {scene.lyrics or '(none)'}
  Section: {scene.notes or '(unspecified)'}"""

    if context_scenes:
        context_lines = []
        for cs in context_scenes:
            context_lines.append(f"  [{cs.id}] {cs.lyrics or '(instrumental)'}")
        user_msg += "\n\nSurrounding scenes for narrative context:\n" + "\n".join(context_lines)

    if scene.characters:
        user_msg += f"\n\nCharacters present in this scene: {', '.join(scene.characters)}"
    if scene.props:
        user_msg += f"\nProps in this scene: {', '.join(scene.props)}"
    if scene.settings:
        user_msg += f"\nSettings in this scene: {', '.join(scene.settings)}"

    user_msg += "\n\nWrite the FLUX image prompt for this scene:"

    log.info("Generating image prompt for %s...", scene.id)
    return client.chat(IMAGE_PROMPT_SYSTEM, user_msg)


def generate_image_prompts_batch(
    scenes: list[Scene],
    config: ProjectConfig,
    llm_config: LLMConfig | None = None,
) -> list[str]:
    """
    Generate prompts for all scenes in one LLM call for better coherence.

    Returns a list of prompt strings in the same order as the input scenes.
    Falls back to individual calls if the batch response can't be parsed.
    """
    client: LLMClient = get_client(llm_config)

    style_context = _build_style_context(config)

    system = IMAGE_PROMPT_SYSTEM + """

When generating multiple scene prompts, output them as a numbered list:
1. <prompt for scene 1>
2. <prompt for scene 2>
...and so on. One prompt per line per scene number. No other text."""

    scene_lines = []
    for i, scene in enumerate(scenes, 1):
        scene_type_label = "instrumental" if scene.type == SceneType.INSTRUMENTAL else "vocal"
        scene_lines.append(
            f"{i}. [{scene.id}] {scene_type_label} | {scene.duration:.1f}s | "
            f"lyrics: {scene.lyrics or '(none)'}"
        )

    user_msg = f"""Style sheet:
{style_context}

Generate a FLUX image prompt for each of the following {len(scenes)} scenes:

{chr(10).join(scene_lines)}"""

    log.info("Generating image prompts for %d scenes (batch)...", len(scenes))
    raw = client.chat(system, user_msg)

    # Parse numbered list
    prompts: list[str] = []
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in lines:
        # Strip leading "N. " or "N) "
        if line and line[0].isdigit():
            dot = line.find(".")
            paren = line.find(")")
            sep = min(x for x in [dot, paren] if x > 0) if (dot > 0 or paren > 0) else -1
            if sep > 0:
                prompts.append(line[sep + 1:].strip())
                continue
        prompts.append(line)

    if len(prompts) != len(scenes):
        log.warning(
            "Batch prompt count mismatch (%d returned, %d expected). "
            "Falling back to individual calls.",
            len(prompts), len(scenes),
        )
        return [
            generate_image_prompt(scene, config, llm_config=llm_config)
            for scene in scenes
        ]

    return prompts
