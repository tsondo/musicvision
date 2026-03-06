"""
Stage 1 orchestrator: Intake & Segmentation.

Two-phase design:
  Phase 1 — run_analyze(): BPM, Whisper, demucs vocal separation (no scenes).
  Phase 2 — create_scenes_from_boundaries(): manual or auto scene boundaries → scenes.json.

run_intake() combines both phases for CLI / backward-compat one-shot usage.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from musicvision.intake.audio_analysis import create_separator, detect_bpm, get_beat_times
from musicvision.intake.segmentation import segment_scenes, segment_scenes_simple
from musicvision.intake.transcription import (
    TranscriptionResult,
    WordTimestamp,
    align_lyrics_with_timestamps,
    load_lyrics_file,
    transcribe,
)
from musicvision.models import (
    AnalysisResult,
    Scene,
    SceneBoundary,
    SceneList,
    SceneType,
    SongSection,
)
from musicvision.project import ProjectService
from musicvision.utils.audio import get_duration, slice_audio
from musicvision.utils.gpu import DeviceMap, detect_devices

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: Analyze (no segmentation)
# ---------------------------------------------------------------------------

def run_analyze(
    project: ProjectService,
    device_map: DeviceMap | None = None,
    skip_transcription: bool = False,
    use_vocal_separation: bool = True,
) -> AnalysisResult:
    """
    Run audio analysis: duration, BPM, beat times, Whisper transcription,
    vocal separation. Does NOT create scenes.

    Results are persisted to project.yaml (BPM, duration, beat_times, sections)
    and input/word_timestamps.json (word-level timestamps).
    """
    if device_map is None:
        device_map = detect_devices()

    config = project.config
    paths = project.paths

    # --- Resolve audio path ---
    if not config.song.audio_file:
        raise ValueError("No audio file set in project config. Upload audio first.")

    audio_path = project.resolve_path(config.song.audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    has_acestep = config.song.acestep is not None

    # --- Duration ---
    log.info("Analyzing audio: %s", audio_path.name)
    if config.song.duration_seconds:
        duration = config.song.duration_seconds
        log.info("Duration (from %s): %.1fs", "AceStep" if has_acestep else "config", duration)
    else:
        duration = get_duration(audio_path)
        config.song.duration_seconds = duration
        log.info("Duration (detected): %.1fs", duration)

    # --- BPM ---
    if config.song.bpm:
        bpm = config.song.bpm
        log.info("BPM (from %s): %.1f", "AceStep" if has_acestep else "config", bpm)
    else:
        bpm = detect_bpm(audio_path)
        config.song.bpm = bpm
        log.info("BPM (detected): %.1f", bpm)

    # --- Beat times ---
    beat_times = get_beat_times(audio_path)
    config.song.beat_times = beat_times
    log.info("Detected %d beats", len(beat_times))

    # --- Vocal separation ---
    vocal_audio_path: Path | None = None
    if use_vocal_separation:
        vocal_full_path = paths.input_dir / "audio_vocal.wav"
        if vocal_full_path.exists():
            log.info("Reusing existing vocal stem: %s", vocal_full_path.name)
            vocal_audio_path = vocal_full_path
        else:
            sep_cfg = config.vocal_separation
            log.info(
                "Running vocal separation (method=%s, model=%s)…",
                sep_cfg.method.value,
                sep_cfg.demucs_model.value,
            )
            separator = create_separator(
                sep_cfg.method,
                device=str(device_map.secondary),
                demucs_model=sep_cfg.demucs_model.value,
            )
            separator.load()
            try:
                vocal_audio_path = separator.separate(
                    audio_path,
                    output_vocal_path=vocal_full_path,
                )
            finally:
                separator.unload()
            log.info("Vocal separation complete")

    whisper_input = vocal_audio_path if vocal_audio_path is not None else audio_path

    # --- Transcription / Lyrics ---
    words: list[WordTimestamp] = []

    lyrics_path = (
        project.resolve_path(config.song.lyrics_file)
        if config.song.lyrics_file
        else None
    )
    has_lyrics = lyrics_path and lyrics_path.exists()

    if skip_transcription and has_lyrics:
        log.info("Using provided lyrics, skipping transcription")
        lyrics_text = load_lyrics_file(lyrics_path)
        words = _approximate_word_timestamps(lyrics_text, duration)

    elif has_lyrics:
        log.info("Transcribing for timestamps, will align with provided lyrics")
        transcription = transcribe(whisper_input, device=str(device_map.secondary))
        lyrics_text = load_lyrics_file(lyrics_path)
        words = align_lyrics_with_timestamps(lyrics_text, transcription)
        log.info("Aligned %d words from provided lyrics", len(words))

    else:
        log.info("No lyrics file — transcribing with Whisper")
        transcription = transcribe(whisper_input, device=str(device_map.secondary))
        words = transcription.words

        lyrics_out = paths.input_dir / "lyrics_whisper.txt"
        lyrics_out.write_text(transcription.text, encoding="utf-8")
        config.song.lyrics_file = "input/lyrics_whisper.txt"
        log.info("Saved Whisper transcription to %s", lyrics_out.name)

    # --- Parse AceStep sections ---
    sections: list[SongSection] = []
    if has_acestep and config.song.acestep.lyrics:
        sections = parse_acestep_sections(config.song.acestep.lyrics, words, duration)
        config.song.sections = sections
        log.info("Parsed %d AceStep sections", len(sections))

    # --- Save word timestamps to disk ---
    word_dicts = [{"word": w.word, "start": w.start, "end": w.end} for w in words]
    ts_path = paths.input_dir / "word_timestamps.json"
    ts_path.write_text(json.dumps(word_dicts, indent=2), encoding="utf-8")
    log.info("Saved %d word timestamps to %s", len(word_dicts), ts_path.name)

    # --- Mark as analyzed ---
    config.song.analyzed = True
    project.config = config
    project.save_config()

    vocal_rel = None
    if vocal_audio_path is not None:
        vocal_rel = str(vocal_audio_path.relative_to(paths.root))

    result = AnalysisResult(
        duration=duration,
        bpm=bpm,
        beat_times=beat_times,
        word_timestamps=word_dicts,
        vocal_path=vocal_rel,
        sections=sections,
    )

    log.info("Analysis complete: %.1fs, %d BPM, %d beats, %d words, %d sections",
             duration, bpm or 0, len(beat_times), len(words), len(sections))

    return result


# ---------------------------------------------------------------------------
# Phase 2: Create scenes from manual boundaries
# ---------------------------------------------------------------------------

def create_scenes_from_boundaries(
    project: ProjectService,
    boundaries: list[SceneBoundary],
    snap_to_beats: bool = False,
) -> SceneList:
    """
    Create scenes from user-provided boundaries. Slices audio per scene.

    Args:
        project: The project to process
        boundaries: Scene time ranges from the waveform editor
        snap_to_beats: Snap boundaries to nearest beat time
    """
    config = project.config
    paths = project.paths

    audio_path = project.resolve_path(config.song.audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    beat_times = config.song.beat_times

    scenes: list[Scene] = []
    for i, b in enumerate(boundaries):
        t_start = b.time_start
        t_end = b.time_end

        if snap_to_beats and beat_times:
            t_start = _snap_to_beat(t_start, beat_times)
            t_end = _snap_to_beat(t_end, beat_times)

        scene_id = f"scene_{i + 1:03d}"
        scene = Scene(
            id=scene_id,
            order=i + 1,
            time_start=t_start,
            time_end=t_end,
            type=b.type,
            lyrics=b.lyrics,
            section=b.section,
        )
        scenes.append(scene)

    scene_list = SceneList(scenes=scenes)

    # Slice audio
    _slice_scenes(scene_list, audio_path, paths)

    # Save
    project.scenes = scene_list
    project.save_scenes()

    log.info("Created %d scenes from manual boundaries (%.1fs total)",
             len(scenes), scene_list.total_duration)

    return scene_list


# ---------------------------------------------------------------------------
# Auto-segment (Phase 2 alternative — LLM/rule-based)
# ---------------------------------------------------------------------------

def run_auto_segment(
    project: ProjectService,
    use_llm: bool = True,
) -> SceneList:
    """
    Run LLM or rule-based segmentation using stored analysis data.
    Requires run_analyze() to have been called first.
    """
    config = project.config
    paths = project.paths

    if not config.song.analyzed:
        raise ValueError("Audio not analyzed yet. Run analyze first.")

    audio_path = project.resolve_path(config.song.audio_file)
    duration = config.song.duration_seconds
    bpm = config.song.bpm
    beat_times = config.song.beat_times

    # Load word timestamps
    ts_path = paths.input_dir / "word_timestamps.json"
    words: list[WordTimestamp] = []
    if ts_path.exists():
        raw = json.loads(ts_path.read_text(encoding="utf-8"))
        words = [WordTimestamp(word=w["word"], start=w["start"], end=w["end"]) for w in raw]

    has_acestep = config.song.acestep is not None
    acestep_caption = config.song.acestep.caption if has_acestep else None
    acestep_lyrics = config.song.acestep.lyrics if has_acestep else None

    # Resolve engine constraints
    engine_constraints = None
    try:
        from musicvision.engine_registry import get_constraints
        engine_constraints = get_constraints(config.video_engine.value)
    except Exception:
        log.debug("Could not resolve engine constraints")

    if use_llm and words:
        log.info("Segmenting with Claude API...")
        scene_list = segment_scenes(
            lyrics_with_timestamps=words,
            song_duration=duration,
            bpm=bpm,
            beat_times=beat_times,
            acestep_caption=acestep_caption,
            acestep_lyrics=acestep_lyrics,
            engine_constraints=engine_constraints,
        )
    else:
        log.info("Segmenting with rule-based splitter...")
        scene_list = segment_scenes_simple(
            lyrics_with_timestamps=words,
            song_duration=duration,
            engine_constraints=engine_constraints,
        )

    # Slice audio
    _slice_scenes(scene_list, audio_path, paths)

    # Save
    project.scenes = scene_list
    project.save_scenes()

    log.info("Auto-segmented into %d scenes (%.1fs total)",
             len(scene_list.scenes), scene_list.total_duration)

    return scene_list


# ---------------------------------------------------------------------------
# Combined one-shot intake (CLI backward compat)
# ---------------------------------------------------------------------------

def run_intake(
    project: ProjectService,
    use_llm_segmentation: bool = True,
    device_map: DeviceMap | None = None,
    skip_transcription: bool = False,
    use_vocal_separation: bool = True,
) -> SceneList:
    """
    Run the full Stage 1 pipeline (analyze + segment in one call).
    This is the CLI entry point. GUI uses run_analyze() + manual boundaries.
    """
    # Phase 1
    run_analyze(
        project=project,
        device_map=device_map,
        skip_transcription=skip_transcription,
        use_vocal_separation=use_vocal_separation,
    )

    # Phase 2
    scene_list = run_auto_segment(
        project=project,
        use_llm=use_llm_segmentation,
    )

    return scene_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_acestep_sections(
    lyrics: str,
    word_timestamps: list[WordTimestamp],
    duration: float,
) -> list[SongSection]:
    """
    Extract section markers from AceStep lyrics (e.g. '(Verse 1)', '(Chorus)').
    Match each to approximate timestamps using word timing data.
    """
    # Find all section markers like (Verse 1), (Hook), (Bridge), (Outro)
    pattern = re.compile(r"\(([^)]+)\)")
    lines = lyrics.split("\n")

    markers: list[tuple[str, int]] = []  # (section_name, line_index)
    for i, line in enumerate(lines):
        m = pattern.match(line.strip())
        if m:
            markers.append((m.group(1), i))

    if not markers:
        return []

    # Build a map: line_index → approximate time
    # Strategy: count non-empty, non-marker lines and distribute word timestamps
    content_lines: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not pattern.match(stripped):
            content_lines.append(i)

    sections: list[SongSection] = []

    if word_timestamps and content_lines:
        # Map content lines to word timestamp positions
        words_per_line = max(1, len(word_timestamps) // max(1, len(content_lines)))

        for name, line_idx in markers:
            # Find how many content lines precede this marker
            preceding = sum(1 for cl in content_lines if cl < line_idx)
            word_idx = min(preceding * words_per_line, len(word_timestamps) - 1)
            t = word_timestamps[word_idx].start if word_idx < len(word_timestamps) else 0.0
            sections.append(SongSection(name=name, time=round(t, 2)))
    else:
        # No word timestamps — distribute evenly
        for i, (name, _) in enumerate(markers):
            t = (i / max(1, len(markers))) * duration
            sections.append(SongSection(name=name, time=round(t, 2)))

    return sections


def _slice_scenes(
    scene_list: SceneList,
    audio_path: Path,
    paths,
) -> None:
    """Slice audio into per-scene segments. Also slices vocal stem if available."""
    log.info("Slicing audio into %d segments...", len(scene_list.scenes))
    for scene in scene_list.scenes:
        segment_path = paths.segment_path(scene.id)
        slice_audio(audio_path, segment_path, scene.time_start, scene.time_end)
        scene.audio_segment = str(segment_path.relative_to(paths.root))

    # Slice vocal stem if available
    vocal_path = paths.input_dir / "audio_vocal.wav"
    if vocal_path.exists():
        log.info("Slicing vocal stem into per-scene segments...")
        for scene in scene_list.scenes:
            vocal_seg = paths.vocal_segment_path(scene.id)
            slice_audio(vocal_path, vocal_seg, scene.time_start, scene.time_end)
            scene.audio_segment_vocal = str(vocal_seg.relative_to(paths.root))


def _snap_to_beat(t: float, beat_times: list[float], tolerance: float = 0.15) -> float:
    """Snap a time to the nearest beat within tolerance."""
    if not beat_times:
        return t
    closest = min(beat_times, key=lambda bt: abs(bt - t))
    if abs(closest - t) <= tolerance:
        return closest
    return t


def _approximate_word_timestamps(lyrics_text: str, duration: float) -> list[WordTimestamp]:
    """
    Create rough word timestamps when we have lyrics but no Whisper.
    Distributes words evenly across the song duration.
    """
    words_raw = lyrics_text.split()
    if not words_raw:
        return []

    word_duration = min(0.3, duration / len(words_raw))
    result: list[WordTimestamp] = []

    for i, word in enumerate(words_raw):
        start = i * word_duration
        end = start + word_duration
        result.append(WordTimestamp(word=word, start=start, end=min(end, duration)))

    return result
