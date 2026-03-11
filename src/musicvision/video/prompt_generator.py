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

import sys

from musicvision.llm import LLMClient, LLMConfig, get_client, llm_available
from musicvision.models import ProjectConfig, Scene, SceneType, StyleSheet

log = logging.getLogger(__name__)

VIDEO_PROMPT_SYSTEM = """You are a music video director writing video generation prompts for HuMo TIA mode.

HuMo TIA (Text + Image + Audio) generates video starting FROM a reference image — \
the first frame is already established. Your prompt should describe what CHANGES from \
that starting point, not re-describe the static scene.

The reference image already establishes composition, lighting, subject placement, and \
environment. Focus your prompt entirely on motion and change.

DO include:
- Physical motion (body movement, gestures, expression shifts, hair/clothing dynamics)
- Camera movement (slow push-in, pan, tracking, static hold with subject motion)
- Environmental changes (wind, light shifts, particle effects, background activity)
- Motion dynamics (speed, direction, acceleration — e.g. "slow drift left to right")
- Subtle changes in lighting or atmosphere over the clip duration

DO NOT include:
- Static scene description already visible in the reference image
- Song or audio references ("as the music swells...")
- Temporal sequencing ("then", "next", "suddenly")
- Abstract emotional language ("hopeful", "melancholic")
- Instructions to the model ("generate", "create", "render")

Length: 2–4 sentences, 100–120 words (HuMo's sweet spot).
Output only the prompt text itself — no commentary, headers, or markdown."""


def _resolve_treatment(scene: Scene, config: ProjectConfig) -> str:
    """Resolve the effective treatment for a scene based on video_type."""
    from musicvision.models import SceneTreatment, VideoType

    if scene.treatment is not None:
        return scene.treatment.value

    vt = config.video_type
    if vt == VideoType.PERFORMANCE:
        return SceneTreatment.PERFORMANCE.value
    if vt == VideoType.STORY:
        return SceneTreatment.NARRATIVE.value
    # Hybrid: default from scene type
    return SceneTreatment.PERFORMANCE.value if scene.type == SceneType.VOCAL else SceneTreatment.NARRATIVE.value


def _treatment_context(treatment: str) -> str:
    """Return hidden prompt context based on scene treatment."""
    if treatment == "performance":
        return (
            "This is a performance scene. The performer is on stage, singing or playing. "
            "Camera captures the live performance energy — body movement, gestures, lip motion."
        )
    return (
        "This is a narrative/cinematic scene. Focus on mood, story, and visual atmosphere. "
        "No performer on stage unless specified. Emphasize environmental motion and cinematic camera work."
    )


def _build_style_context(config: ProjectConfig, scene: Scene | None = None) -> str:
    """Serialize the style sheet into a compact text block for the LLM."""
    ss = config.style_sheet
    parts: list[str] = []

    if ss.concept:
        parts.append(f"Video concept: {ss.concept}")

    # Inject treatment context if scene is provided
    if scene is not None:
        treatment = _resolve_treatment(scene, config)
        parts.append(f"Scene treatment: {_treatment_context(treatment)}")

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
    if not llm_available(llm_config):
        log.warning("LLM unavailable — falling back to interactive input for %s", scene.id)
        return _prompt_interactive_video(scene, config)

    client: LLMClient = get_client(llm_config)

    style_context = _build_style_context(config, scene)

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
    try:
        return client.chat(VIDEO_PROMPT_SYSTEM, user_msg)
    except Exception as exc:
        log.warning("LLM call failed for %s: %s — falling back", scene.id, exc)
        return _prompt_interactive_video(scene, config)


def generate_video_prompts_batch(
    scenes: list[Scene],
    config: ProjectConfig,
    llm_config: LLMConfig | None = None,
) -> list[str]:
    """
    Generate video prompts for all scenes in one LLM call for better coherence.

    Returns a list of prompt strings in the same order as the input scenes.
    Falls back to individual calls if the batch response can't be parsed.
    """
    import json

    client: LLMClient = get_client(llm_config)

    style_context = _build_style_context(config)

    system = VIDEO_PROMPT_SYSTEM + """

When generating multiple scene prompts, output them as a JSON array of strings:
["prompt for scene 1", "prompt for scene 2", ...]
One string per scene, in order. Output ONLY the JSON array — no other text."""

    scene_lines = []
    for i, scene in enumerate(scenes, 1):
        scene_type_label = "instrumental" if scene.type == SceneType.INSTRUMENTAL else "vocal"
        treatment = _resolve_treatment(scene, config)
        treatment_hint = _treatment_context(treatment)
        img_hint = f" | ref image: {scene.effective_image_prompt[:100]}" if scene.effective_image_prompt else ""
        scene_lines.append(
            f"{i}. [{scene.id}] {scene_type_label} ({treatment}) | {scene.duration:.1f}s | "
            f"lyrics: {scene.lyrics or '(none)'} | {treatment_hint}{img_hint}"
        )

    user_msg = f"""Style sheet:
{style_context}

Generate a HuMo video prompt for each of the following {len(scenes)} scenes:

{chr(10).join(scene_lines)}"""

    log.info("Generating video prompts for %d scenes (batch)...", len(scenes))
    raw = client.chat(system, user_msg)

    # Strip markdown code block fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Parse JSON array
    try:
        prompts = json.loads(raw)
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError("Expected a JSON array of strings")
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning(
            "Batch video prompt JSON parse failed (%s). Falling back to individual calls.", exc,
        )
        return [
            generate_video_prompt(scene, config, llm_config=llm_config)
            for scene in scenes
        ]

    if len(prompts) != len(scenes):
        log.warning(
            "Batch returned %d prompts for %d scenes — falling back to individual calls.",
            len(prompts), len(scenes),
        )
        return [
            generate_video_prompt(scene, config, llm_config=llm_config)
            for scene in scenes
        ]

    return prompts


# ---------------------------------------------------------------------------
# Interactive fallback
# ---------------------------------------------------------------------------

def _prompt_interactive_video(scene: Scene, config: ProjectConfig) -> str:
    """
    Prompt the user to type a HuMo video prompt when no LLM is available.

    If stdin is not a TTY (non-interactive / piped), returns a minimal
    auto-generated template instead of blocking.
    """
    style_context = _build_style_context(config)
    scene_type_label = "instrumental" if scene.type == SceneType.INSTRUMENTAL else "vocal"
    divider = "─" * 60

    if not sys.stdin.isatty():
        template = _auto_template_video(scene, config)
        log.info("Non-interactive mode — using auto-template for %s: %s", scene.id, template)
        return template

    print(f"\n{divider}")
    print(f"LLM unavailable — video prompt required for {scene.id}")
    print(divider)
    print(f"Scene:    {scene.id}  |  {scene_type_label}  |  {scene.duration:.1f}s")
    if scene.lyrics:
        snippet = scene.lyrics[:120] + ("…" if len(scene.lyrics) > 120 else "")
        print(f"Lyrics:   {snippet}")
    if scene.effective_image_prompt:
        img_snippet = scene.effective_image_prompt[:120] + (
            "…" if len(scene.effective_image_prompt) > 120 else ""
        )
        print(f"Image:    {img_snippet}")
    if style_context != "(no style sheet defined)":
        print(f"Style:    {style_context}")
    print()
    print("Guidelines: Dense visual + motion description, 3–5 sentences, 80–160 words.")
    print("  Include:  appearance, environment, lighting, motion, camera framing")
    print("  Avoid:    abstract emotion, audio references, 'then'/'next'")
    print()
    print("Enter HuMo video prompt (or press Enter for auto-template):")

    try:
        response = input("> ").strip()
    except EOFError:
        response = ""

    if not response:
        response = _auto_template_video(scene, config)
        print(f"  [auto-template] {response}")

    return response


def _auto_template_video(scene: Scene, config: ProjectConfig) -> str:
    """Minimal template used when interactive input is unavailable or skipped."""
    ss = config.style_sheet
    style = ss.visual_style or "cinematic music video"
    subject = ss.characters[0].description if ss.characters else "a performer"
    setting = ss.settings[0].description if ss.settings else "a dramatic stage"
    motion = "moving with the rhythm, gesturing expressively" if scene.type == SceneType.VOCAL \
             else "standing still, ambient environmental motion"
    lyric_hint = f'with lyrics "{scene.lyrics[:60]}"' if scene.lyrics else "instrumental passage"
    # Prefer the image prompt as the visual anchor if available
    anchor = (
        f"The scene begins from the reference image: {scene.effective_image_prompt[:80]}. "
        if scene.effective_image_prompt else ""
    )
    return (
        f"{anchor}{subject.capitalize()} in {setting}, {style} aesthetic. "
        f"Medium close-up shot, camera slowly pushing in, {motion}. "
        f"Dramatic side lighting with deep shadows, {ss.color_palette or 'rich colors'}. "
        f"Scene corresponds to {lyric_hint}."
    )
