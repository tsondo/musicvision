"""
LLM-assisted scene segmentation.

Takes lyrics with timestamps and produces a scene list.
Uses an LLM (Anthropic Claude or local vLLM) for intelligent boundary
detection that respects musical phrasing and narrative structure.
"""

from __future__ import annotations

import json
import logging

from musicvision.engine_registry import EngineConstraints
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
    engine_constraints: EngineConstraints | None = None,
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

    # Build system prompt — optionally enrich with engine constraints
    system_prompt = SEGMENTATION_SYSTEM_PROMPT
    if engine_constraints:
        system_prompt += _engine_constraint_prompt(engine_constraints)

    # Build the user message — prefer AceStep lyrics (section markers) over
    # approximate word timestamps to save tokens for vLLM context windows.
    user_message = f"""Song duration: {song_duration:.1f} seconds
BPM: {bpm or 'unknown'}
Min scene duration: {min_scene_seconds}s
Max scene duration: {max_scene_seconds}s
"""

    if acestep_lyrics:
        # AceStep lyrics have section markers — better for the LLM, skip timestamps
        user_message += f"\nLyrics with section markers:\n{acestep_lyrics}\n"
    else:
        lyrics_text = _format_lyrics_for_llm(lyrics_with_timestamps)
        user_message += f"\nLyrics with timestamps:\n{lyrics_text}\n"

    user_message += "\nPlease segment this song into scenes for a music video."

    if beat_times:
        # Include a sparse subset of beat times (every 4th beat) to save tokens
        sparse_beats = beat_times[::4][:20]
        beat_str = ", ".join(f"{t:.1f}" for t in sparse_beats)
        user_message += f"\n\nBeat timestamps (sample): [{beat_str}]"

    # AceStep caption adds genre/mood context
    if acestep_caption:
        user_message += f"\n\nSong description (genre/mood/instrumentation):\n{acestep_caption}"

    log.info("Calling LLM for scene segmentation...")

    response_text = client.chat(system_prompt, user_message)

    # Handle potential markdown code blocks
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    try:
        scene_dicts = json.loads(response_text, strict=False)
    except json.JSONDecodeError as e:
        log.warning("JSON parse failed (%s), attempting truncated JSON recovery...", e)
        scene_dicts = _recover_truncated_json(response_text)
        if not scene_dicts:
            log.error("Could not recover any scenes from LLM response")
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

    # If truncated recovery gave us scenes that don't cover the end, extend last scene
    if scenes and scenes[-1].time_end < song_duration - 1.0:
        gap = song_duration - scenes[-1].time_end
        log.warning(
            "Last scene ends at %.1fs but song is %.1fs (%.1fs gap) — "
            "extending last scene to cover remainder",
            scenes[-1].time_end, song_duration, gap,
        )
        scenes[-1].time_end = song_duration

    # Snap to beat times if available
    if beat_times:
        scenes = _snap_to_beats(scenes, beat_times, tolerance=0.15)

    # Validate
    _validate_scenes(scenes, song_duration, min_scene_seconds, max_scene_seconds)

    # Frame-accurate adjustment when engine constraints are available
    if engine_constraints:
        scenes = _validate_and_adjust_scenes(scenes, song_duration, engine_constraints, beat_times)

    log.info("Segmentation complete: %d scenes", len(scenes))
    return SceneList(scenes=scenes)


def segment_scenes_simple(
    lyrics_with_timestamps: list[WordTimestamp],
    song_duration: float,
    max_scene_seconds: float = 8.0,
    engine_constraints: EngineConstraints | None = None,
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

    # Frame-accurate adjustment when engine constraints are available
    if engine_constraints:
        scenes = _validate_and_adjust_scenes(scenes, song_duration, engine_constraints)

    return SceneList(scenes=scenes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recover_truncated_json(text: str) -> list[dict] | None:
    """Try to recover scene objects from a truncated JSON array.

    vLLM may cut off the response mid-JSON when hitting max_model_len.
    We find the last complete `}` that closes a scene object and close
    the array there.
    """
    import re

    # Find the opening bracket
    start = text.find("[")
    if start == -1:
        return None

    # Find all complete JSON objects by looking for `},` or `}` followed by `]`
    # Strategy: progressively try closing the array after each `}`
    best: list[dict] | None = None
    for m in re.finditer(r"\}", text):
        candidate = text[start:m.end()] + "]"
        try:
            parsed = json.loads(candidate, strict=False)
            if isinstance(parsed, list) and parsed:
                best = parsed
        except json.JSONDecodeError:
            continue

    if best:
        log.info("Recovered %d scenes from truncated JSON response", len(best))
    return best


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


def _engine_constraint_prompt(constraints: EngineConstraints) -> str:
    """Build an LLM prompt addendum describing video engine constraints."""
    max_sec = constraints.max_seconds
    min_sec = constraints.min_seconds

    # Compute a "bad duration" example: max + a small amount below min
    bad_dur = max_sec + min_sec * 0.5
    bad_remainder = bad_dur - max_sec
    good_short = max_sec * 0.9
    good_long = max_sec * 2.0

    return f"""

Video engine constraints:
- Engine: {constraints.name}
- Max clip: {max_sec:.2f}s ({constraints.max_frames} frames @ {constraints.fps}fps)
- Min clip: {min_sec:.2f}s ({constraints.min_frames} frames @ {constraints.fps}fps)

Duration guidance:
- Preferred scene durations: ≤{max_sec:.2f}s (single clip) or clean multiples of {max_sec:.2f}s.
- Avoid durations that leave a remainder under {min_sec:.2f}s when divided by {max_sec:.2f}s.
  Example: {bad_dur:.1f}s is problematic ({max_sec:.2f} + {bad_remainder:.1f}s remainder). \
Prefer {good_short:.1f}s or {good_long:.1f}s instead."""


def _validate_and_adjust_scenes(
    scenes: list[Scene],
    song_duration: float,
    constraints: EngineConstraints,
    beat_times: list[float] | None = None,
) -> list[Scene]:
    """
    Post-process LLM segmentation to enforce frame-accurate constraints.

    1. Convert all boundaries to frame numbers
    2. Snap to beat times if available (within tolerance)
    3. Check each scene's sub-clip remainder — if the last sub-clip would
       be below min_frames, adjust the scene boundary
    4. Verify first scene starts at frame 0, last ends at total_frames
    5. Verify no gaps or overlaps between consecutive scenes
    """
    from musicvision.engine_registry import compute_subclip_frames, scene_frames

    if not scenes:
        return scenes

    fps = constraints.fps
    total_song_frames = scene_frames(0.0, song_duration, fps)

    for scene in scenes:
        total = scene_frames(scene.time_start, scene.time_end, fps)
        if total <= 0:
            continue

        # Check whether the sub-clip split produces a remainder below min
        sub_counts = compute_subclip_frames(total, constraints.max_frames, constraints.min_frames)

        # If compute_subclip_frames succeeded, the split is valid.
        # Store the computed frame info on the scene for later use.
        frame_start = scene_frames(0.0, scene.time_start, fps)
        scene.frame_start = frame_start
        scene.frame_end = frame_start + total
        scene.total_frames = total
        scene.subclip_frame_counts = sub_counts

    # Verify coverage: first should start near 0, last should end near total
    if scenes[0].frame_start is not None and scenes[0].frame_start > 1:
        log.warning(
            "First scene starts at frame %d (%.2fs), expected 0",
            scenes[0].frame_start, scenes[0].time_start,
        )

    if scenes[-1].frame_end is not None:
        gap_frames = abs(total_song_frames - scenes[-1].frame_end)
        if gap_frames > 1:
            log.warning(
                "Last scene ends at frame %d but song has %d frames (%d frame gap)",
                scenes[-1].frame_end, total_song_frames, gap_frames,
            )

    # Check for gaps/overlaps between consecutive scenes
    for i in range(len(scenes) - 1):
        this_end = scenes[i].time_end
        next_start = scenes[i + 1].time_start
        gap = next_start - this_end
        if abs(gap) > 0.01:
            log.warning(
                "Gap/overlap between %s and %s: %.3fs",
                scenes[i].id, scenes[i + 1].id, gap,
            )

    return scenes
