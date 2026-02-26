"""
LLM-assisted scene segmentation.

Takes lyrics with timestamps and produces a scene list.
Uses an LLM (Anthropic Claude or local vLLM) for intelligent boundary
detection that respects musical phrasing and narrative structure.
"""

from __future__ import annotations

import json
import logging

from musicvision.intake.transcription import WordTimestamp
from musicvision.llm import LLMClient, LLMConfig, get_client
from musicvision.models import Scene, SceneList, SceneType

log = logging.getLogger(__name__)

# System prompt for scene segmentation
SEGMENTATION_SYSTEM_PROMPT = """You are a music video director segmenting a song into visual scenes.

Given lyrics with word-level timestamps, produce a JSON list of scenes.

Rules:
- Each scene has a time_start and time_end in seconds
- Minimum scene duration: 2.0 seconds
- Maximum scene duration: 10.0 seconds
- Prefer cutting on phrase/line boundaries — don't split mid-word or mid-phrase
- Consecutive lines that form one thought can be one scene
- Instrumental gaps (no lyrics) get their own scene with type "instrumental"
- The first scene starts at 0.0
- The last scene ends at the song's total duration
- Verse/chorus/bridge sections should be noted
- Chorus repetitions should be marked so imagery can be reused with variations
- Number scenes sequentially: scene_001, scene_002, etc.

Section markers (AceStep):
- When the lyrics include section markers like (Verse 1), (Hook), (Bridge), (Outro), \
use them directly as the "section" field values (e.g. "Verse 1", "Hook", "Bridge"). \
Do NOT invent new section labels — mirror the markers from the input exactly.

Section boundary rules:
- Scenes must NOT cross section boundaries. If a section boundary falls within a \
scene's time range, split the scene at that boundary (respecting the minimum duration).
- Instrumental intros, outros, and instrumental breaks between vocal sections should \
each be their own scene with type "instrumental".

Output format (JSON array):
[
  {
    "id": "scene_001",
    "order": 1,
    "time_start": 0.0,
    "time_end": 3.5,
    "type": "instrumental",
    "lyrics": "",
    "section": "intro"
  },
  {
    "id": "scene_002",
    "order": 2,
    "time_start": 3.5,
    "time_end": 7.2,
    "type": "vocal",
    "lyrics": "Standing in the rain tonight",
    "section": "verse_1"
  }
]

Respond with ONLY the JSON array, no other text."""


def segment_scenes(
    lyrics_with_timestamps: list[WordTimestamp],
    song_duration: float,
    bpm: float | None = None,
    beat_times: list[float] | None = None,
    min_scene_seconds: float = 2.0,
    max_scene_seconds: float = 10.0,
    acestep_caption: str | None = None,
    acestep_lyrics: str | None = None,
    api_key: str | None = None,
    llm_config: LLMConfig | None = None,
) -> SceneList:
    """
    Segment a song into scenes using an LLM.

    Backend is controlled by LLM_BACKEND env var ('anthropic' or 'openai').
    See musicvision/llm.py for full configuration details.

    Args:
        lyrics_with_timestamps: Word-level timestamps from Whisper
        song_duration: Total song duration in seconds
        bpm: Detected BPM (informational for the LLM)
        beat_times: Beat timestamps for boundary snapping
        min_scene_seconds: Minimum scene duration
        max_scene_seconds: Maximum scene duration
        api_key: Anthropic API key (legacy; prefer LLM_BACKEND + ANTHROPIC_API_KEY env vars)
        llm_config: Explicit LLMConfig override; if None, reads from environment

    Returns:
        SceneList with timestamped scenes
    """
    # Legacy api_key param: wrap into an explicit Anthropic config so callers
    # that pass api_key= directly still work without any env var changes.
    if llm_config is None and api_key:
        llm_config = LLMConfig(backend="anthropic", api_key=api_key)

    client: LLMClient = get_client(llm_config)

    # Build the user message with all context
    lyrics_text = _format_lyrics_for_llm(lyrics_with_timestamps)

    user_message = f"""Song duration: {song_duration:.1f} seconds
BPM: {bpm or 'unknown'}
Min scene duration: {min_scene_seconds}s
Max scene duration: {max_scene_seconds}s

Lyrics with timestamps:
{lyrics_text}

Please segment this song into scenes for a music video."""

    if beat_times:
        # Include a subset of beat times to help with boundary decisions
        beat_str = ", ".join(f"{t:.2f}" for t in beat_times[:50])
        user_message += f"\n\nBeat timestamps (first 50): [{beat_str}]"

    # AceStep provides section markers in lyrics and a genre/mood caption
    if acestep_lyrics:
        user_message += f"\n\nOriginal lyrics with section markers (from AceStep):\n{acestep_lyrics}"
    if acestep_caption:
        user_message += f"\n\nSong description (genre/mood/instrumentation):\n{acestep_caption}"

    log.info("Calling LLM for scene segmentation...")

    response_text = client.chat(SEGMENTATION_SYSTEM_PROMPT, user_message)

    # Handle potential markdown code blocks
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    try:
        scene_dicts = json.loads(response_text)
    except json.JSONDecodeError as e:
        log.error("Failed to parse LLM response as JSON: %s", e)
        log.error("Response was: %s", response_text[:500])
        raise ValueError(f"LLM returned invalid JSON: {e}")

    # Convert to Scene objects
    scenes = []
    for sd in scene_dicts:
        scene = Scene(
            id=sd["id"],
            order=sd["order"],
            time_start=float(sd["time_start"]),
            time_end=float(sd["time_end"]),
            type=SceneType(sd.get("type", "vocal")),
            lyrics=sd.get("lyrics", ""),
        )
        scenes.append(scene)

    # Snap to beat times if available
    if beat_times:
        scenes = _snap_to_beats(scenes, beat_times, tolerance=0.15)

    # Validate
    _validate_scenes(scenes, song_duration, min_scene_seconds, max_scene_seconds)

    log.info("Segmentation complete: %d scenes", len(scenes))
    return SceneList(scenes=scenes)


def segment_scenes_simple(
    lyrics_with_timestamps: list[WordTimestamp],
    song_duration: float,
    max_scene_seconds: float = 8.0,
) -> SceneList:
    """
    Simple rule-based segmentation — no LLM needed.

    Splits on line breaks (gaps > 0.3s between words) with a max duration cap.
    Useful as a fallback or for quick iteration without API calls.
    """
    if not lyrics_with_timestamps:
        # No lyrics — single instrumental scene
        return SceneList(scenes=[
            Scene(id="scene_001", order=1, time_start=0.0, time_end=song_duration,
                  type=SceneType.INSTRUMENTAL, lyrics=""),
        ])

    scenes: list[Scene] = []
    current_words: list[WordTimestamp] = []
    scene_start = 0.0
    scene_num = 1

    for i, word in enumerate(lyrics_with_timestamps):
        # Detect gap before this word
        gap = word.start - (current_words[-1].end if current_words else scene_start)
        duration_so_far = word.end - scene_start

        should_split = (
            (gap > 0.5 and duration_so_far > 2.0)  # natural pause
            or duration_so_far > max_scene_seconds    # max duration
        )

        if should_split and current_words:
            # Close current scene
            scene_end = current_words[-1].end
            lyrics = " ".join(w.word for w in current_words)

            # Check for instrumental gap before this scene
            if scene_start < current_words[0].start - 1.0:
                # Insert instrumental scene for the gap
                scenes.append(Scene(
                    id=f"scene_{scene_num:03d}",
                    order=scene_num,
                    time_start=scene_start,
                    time_end=current_words[0].start,
                    type=SceneType.INSTRUMENTAL,
                    lyrics="",
                ))
                scene_num += 1
                scene_start = current_words[0].start

            scenes.append(Scene(
                id=f"scene_{scene_num:03d}",
                order=scene_num,
                time_start=scene_start,
                time_end=scene_end,
                type=SceneType.VOCAL,
                lyrics=lyrics,
            ))
            scene_num += 1
            scene_start = scene_end
            current_words = [word]
        else:
            current_words.append(word)

    # Final scene
    if current_words:
        lyrics = " ".join(w.word for w in current_words)
        scenes.append(Scene(
            id=f"scene_{scene_num:03d}",
            order=scene_num,
            time_start=scene_start,
            time_end=current_words[-1].end,
            type=SceneType.VOCAL,
            lyrics=lyrics,
        ))
        scene_num += 1

    # Trailing instrumental
    last_end = scenes[-1].time_end if scenes else 0.0
    if song_duration - last_end > 1.0:
        scenes.append(Scene(
            id=f"scene_{scene_num:03d}",
            order=scene_num,
            time_start=last_end,
            time_end=song_duration,
            type=SceneType.INSTRUMENTAL,
            lyrics="",
        ))

    return SceneList(scenes=scenes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_lyrics_for_llm(words: list[WordTimestamp]) -> str:
    """Format word timestamps as readable text for the LLM."""
    lines: list[str] = []
    current_line_words: list[str] = []
    last_end = 0.0
    line_start = 0.0

    for w in words:
        gap = w.start - last_end

        if gap > 0.5 and current_line_words:
            # Line break
            line_text = " ".join(current_line_words)
            lines.append(f"[{line_start:.2f}s - {last_end:.2f}s] {line_text}")
            current_line_words = [w.word]
            line_start = w.start
        else:
            if not current_line_words:
                line_start = w.start
            current_line_words.append(w.word)

        last_end = w.end

    if current_line_words:
        line_text = " ".join(current_line_words)
        lines.append(f"[{line_start:.2f}s - {last_end:.2f}s] {line_text}")

    return "\n".join(lines)


def _snap_to_beats(
    scenes: list[Scene],
    beat_times: list[float],
    tolerance: float = 0.15,
) -> list[Scene]:
    """Snap scene boundaries to nearest beat if within tolerance."""
    import numpy as np

    beats = np.array(beat_times)

    for scene in scenes:
        # Snap start
        diffs = np.abs(beats - scene.time_start)
        nearest_idx = diffs.argmin()
        if diffs[nearest_idx] < tolerance:
            scene.time_start = float(beats[nearest_idx])

        # Snap end
        diffs = np.abs(beats - scene.time_end)
        nearest_idx = diffs.argmin()
        if diffs[nearest_idx] < tolerance:
            scene.time_end = float(beats[nearest_idx])

    return scenes


def _validate_scenes(
    scenes: list[Scene],
    song_duration: float,
    min_duration: float,
    max_duration: float,
) -> None:
    """Log warnings for scenes that violate constraints."""
    for scene in scenes:
        d = scene.duration
        if d < min_duration:
            log.warning("Scene %s is too short: %.2fs (min %.1fs)", scene.id, d, min_duration)
        if d > max_duration:
            log.warning("Scene %s is too long: %.2fs (max %.1fs)", scene.id, d, max_duration)

    if scenes:
        if scenes[0].time_start > 0.5:
            log.warning("Gap at start: first scene begins at %.2fs", scenes[0].time_start)
        if abs(scenes[-1].time_end - song_duration) > 0.5:
            log.warning(
                "Gap at end: last scene ends at %.2fs but song is %.2fs",
                scenes[-1].time_end, song_duration,
            )
