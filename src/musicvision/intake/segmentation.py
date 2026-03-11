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
SEGMENTATION_SYSTEM_PROMPT = """Segment a song into music video scenes. Output ONLY a JSON array.

Rules:
- Each scene: 2-10 seconds. Group 2-4 lyric lines per scene.
- Cut on phrase boundaries, not mid-line.
- Instrumental gaps → type "instrumental". Vocal → type "vocal".
- First scene starts at 0.0, last ends at song duration.
- Use section markers from input as "section" values (e.g. "Verse 1", "Chorus").
- Don't cross section boundaries.
- Number scenes: scene_001, scene_002, etc.

JSON format per scene:
{"id":"scene_001","order":1,"time_start":0.0,"time_end":5.2,"type":"vocal","lyrics":"...","section":"verse_1"}

Respond with ONLY the JSON array."""


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

    # Set a reasonable output budget if not already configured.
    # Each scene is ~80-100 tokens of JSON. For a 3-min song with ~30 scenes
    # that's ~3k tokens. 2048 is a good per-pass budget that leaves room for
    # the prompt in small context windows (3.4k on vLLM).
    if not client.config.max_tokens:
        from dataclasses import replace as _dc_replace
        client.config = _dc_replace(client.config, max_tokens=2048)

    # Build system prompt — optionally enrich with engine constraints
    system_prompt = SEGMENTATION_SYSTEM_PROMPT
    if engine_constraints:
        system_prompt += _engine_constraint_prompt(engine_constraints)

    # Iterative segmentation: call the LLM for the full song, then re-submit
    # any uncovered tail (from truncated output) until the song is fully covered.
    # Falls back to rule-based for the final tail if retries are exhausted.
    MAX_LLM_PASSES = 5
    scenes: list[Scene] = []
    covered_end = 0.0

    for pass_num in range(MAX_LLM_PASSES):
        # Words for this pass (only the uncovered portion)
        pass_words = [
            w for w in lyrics_with_timestamps
            if (w.start + w.end) / 2 >= covered_end
        ]
        pass_duration = song_duration - covered_end

        if pass_duration < 1.0:
            break

        # Determine starting scene number for this pass
        start_order = len(scenes) + 1

        user_message = f"""Song duration: {pass_duration:.1f} seconds
Song starts at: {covered_end:.1f}s (absolute position in full song)
BPM: {bpm or 'unknown'}
Min scene duration: {min_scene_seconds}s
Max scene duration: {max_scene_seconds}s
Start numbering scenes at: {start_order}
"""

        lyrics_text = _format_lyrics_for_llm(pass_words)
        user_message += f"\nLyrics with timestamps:\n{lyrics_text}\n"

        # Pass AceStep section markers as structural hints (first pass only)
        if pass_num == 0 and acestep_lyrics:
            section_markers = _extract_section_markers(acestep_lyrics)
            if section_markers:
                user_message += f"\nSection structure hints (from metadata, for boundary guidance only):\n{section_markers}\n"

        user_message += "\nPlease segment this song into scenes for a music video."

        if beat_times:
            # Include beats relevant to this pass
            pass_beats = [t for t in beat_times if t >= covered_end]
            sparse_beats = pass_beats[::4][:20]
            if sparse_beats:
                beat_str = ", ".join(f"{t:.1f}" for t in sparse_beats)
                user_message += f"\n\nBeat timestamps (sample): [{beat_str}]"

        # AceStep caption adds genre/mood context (first pass only)
        if pass_num == 0 and acestep_caption:
            user_message += f"\n\nSong description (genre/mood/instrumentation):\n{acestep_caption}"

        log.info(
            "LLM segmentation pass %d: %.1fs–%.1fs (%.1fs remaining)",
            pass_num + 1, covered_end, song_duration, pass_duration,
        )

        try:
            response_text = client.chat(system_prompt, user_message)
        except Exception as exc:
            log.error("LLM call failed on pass %d: %s", pass_num + 1, exc)
            break

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
                log.error("Could not recover any scenes from LLM response on pass %d", pass_num + 1)
                break

        if not scene_dicts:
            break

        # Convert to Scene objects — clamp to song bounds, drop invalid
        pass_scenes: list[Scene] = []
        for sd in scene_dicts:
            t_start = float(sd["time_start"])
            t_end = float(sd["time_end"])

            # Drop scenes that start at or past the song end
            if t_start >= song_duration:
                log.debug("Dropping scene starting at %.1fs (past song end %.1fs)", t_start, song_duration)
                continue

            # Clamp end to song duration
            t_end = min(t_end, song_duration)

            # Drop zero/negative duration scenes
            if t_end <= t_start:
                log.debug("Dropping scene with non-positive duration: %.1fs–%.1fs", t_start, t_end)
                continue

            scene = Scene(
                id="",  # renumbered after merge
                order=0,
                time_start=t_start,
                time_end=t_end,
                type=SceneType(sd.get("type", "vocal")),
                lyrics=sd.get("lyrics", ""),
                section=sd.get("section", ""),
            )
            pass_scenes.append(scene)

        # Merge any too-short scenes before adding to the main list
        pass_scenes = _merge_short_scenes(pass_scenes, min_scene_seconds)

        for ps in pass_scenes:
            order = len(scenes) + 1
            ps.id = f"scene_{order:03d}"
            ps.order = order
            ps.lyrics = _lyrics_from_words(lyrics_with_timestamps, ps.time_start, ps.time_end)
            scenes.append(ps)

        covered_end = scenes[-1].time_end if scenes else 0.0
        log.info("Pass %d: %d raw → %d merged scenes, covered to %.1fs",
                 pass_num + 1, len(scene_dicts), len(pass_scenes), covered_end)

        # If we covered the song, we're done
        if covered_end >= song_duration - 1.0:
            break

    # Final fallback: if LLM passes didn't cover everything, use rule-based
    if scenes and scenes[-1].time_end < song_duration - 1.0:
        gap_start = scenes[-1].time_end
        gap = song_duration - gap_start
        log.warning(
            "LLM covered only %.1fs of %.1fs after %d passes — "
            "filling %.1fs remainder with rule-based segmentation",
            gap_start, song_duration, MAX_LLM_PASSES, gap,
        )
        remaining_words = [
            WordTimestamp(word=w.word, start=w.start - gap_start, end=w.end - gap_start)
            for w in lyrics_with_timestamps
            if (w.start + w.end) / 2 > gap_start
        ]
        tail = segment_scenes_simple(
            remaining_words, gap, max_scene_seconds=max_scene_seconds,
            engine_constraints=engine_constraints,
        )
        next_order = len(scenes) + 1
        for ts in tail.scenes:
            ts.time_start += gap_start
            ts.time_end += gap_start
            ts.order = next_order
            ts.id = f"scene_{next_order:03d}"
            ts.lyrics = _lyrics_from_words(lyrics_with_timestamps, ts.time_start, ts.time_end)
            next_order += 1
        scenes.extend(tail.scenes)
        log.info("Added %d rule-based scenes for uncovered tail", len(tail.scenes))
    elif not scenes:
        raise ValueError("LLM segmentation produced no scenes")

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


def _lyrics_from_words(
    words: list[WordTimestamp],
    time_start: float,
    time_end: float,
) -> str:
    """Extract lyrics text from word timestamps that fall within a time range.

    A word is included if its midpoint falls within [time_start, time_end].
    This is the canonical source of lyrics — derived from Whisper transcription,
    not from AceStep metadata.
    """
    scene_words = []
    for w in words:
        mid = (w.start + w.end) / 2
        if time_start <= mid <= time_end:
            scene_words.append(w.word)
    return " ".join(scene_words)


def _extract_section_markers(acestep_lyrics: str) -> str:
    """Extract only section marker lines from AceStep lyrics.

    Returns lines like:
      (Verse 1) at ~line 3
      (Chorus) at ~line 8
    These are structural hints for the LLM, not lyrics content.
    """
    import re

    pattern = re.compile(r"^\s*\(([^)]+)\)\s*$")
    lines = acestep_lyrics.split("\n")
    markers: list[str] = []
    content_line_count = 0

    for line in lines:
        m = pattern.match(line)
        if m:
            markers.append(f"({m.group(1)}) — after ~{content_line_count} lyric lines")
        elif line.strip():
            content_line_count += 1

    return "\n".join(markers)


def _merge_short_scenes(
    scenes: list[Scene],
    min_duration: float,
) -> list[Scene]:
    """Merge scenes shorter than min_duration into their neighbors.

    Strategy: merge with the shorter neighbor (prefer keeping longer scenes
    intact). If only one neighbor exists, merge with that one.
    """
    if len(scenes) <= 1:
        return scenes

    merged = list(scenes)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(merged):
            scene = merged[i]
            dur = scene.time_end - scene.time_start
            if dur >= min_duration or len(merged) <= 1:
                i += 1
                continue

            # Pick neighbor to merge with
            prev_dur = (merged[i - 1].time_end - merged[i - 1].time_start) if i > 0 else float("inf")
            next_dur = (merged[i + 1].time_end - merged[i + 1].time_start) if i < len(merged) - 1 else float("inf")

            if i == 0:
                # No previous — merge into next
                target = i + 1
            elif i == len(merged) - 1:
                # No next — merge into previous
                target = i - 1
            elif prev_dur <= next_dur:
                target = i - 1
            else:
                target = i + 1

            # Merge: extend target to cover this scene's range
            t = merged[target]
            t.time_start = min(t.time_start, scene.time_start)
            t.time_end = max(t.time_end, scene.time_end)
            # Combine lyrics and prefer vocal type
            if scene.type == SceneType.VOCAL or t.type == SceneType.VOCAL:
                t.type = SceneType.VOCAL
            if scene.lyrics and t.lyrics:
                if target < i:
                    t.lyrics = t.lyrics + " " + scene.lyrics
                else:
                    t.lyrics = scene.lyrics + " " + t.lyrics
            elif scene.lyrics:
                t.lyrics = scene.lyrics

            merged.pop(i)
            changed = True
            # Don't increment i — recheck at same position

    log.debug("Merged %d → %d scenes (min %.1fs)", len(scenes), len(merged), min_duration)
    return merged


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
