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

from musicvision.intake.audio_analysis import create_separator, detect_bpm, detect_sections, get_beat_times
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

    warnings: list[str] = []
    if duration > 360:
        msg = (
            f"Song is {duration / 60:.1f} minutes — songs over 6 minutes will produce "
            "many scenes and may take significantly longer to process"
        )
        log.warning(msg)
        warnings.append(msg)

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
        words = _approximate_word_timestamps(lyrics_text, duration, bpm=config.song.bpm)

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

    # --- Parse AceStep sections or auto-detect from audio ---
    sections: list[SongSection] = []
    if has_acestep and config.song.acestep.lyrics:
        sections = parse_acestep_sections(config.song.acestep.lyrics, words, duration)
        config.song.sections = sections
        config.song.sections_source = "acestep"
        log.info("Parsed %d AceStep sections", len(sections))
    else:
        # No AceStep metadata — auto-detect sections from audio features
        try:
            detected = detect_sections(audio_path, beat_times, duration)
            sections = [SongSection(name=name, time=t) for name, t in detected]
            config.song.sections = sections
            config.song.sections_source = "auto"
            log.info("Auto-detected %d sections from audio (no AceStep metadata)", len(sections))
        except Exception:
            log.warning("Section auto-detection failed; continuing without sections", exc_info=True)

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
        warnings=warnings,
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
    device: str | None = None,
) -> SceneList:
    """
    Create scenes from user-provided boundaries. Slices audio per scene,
    then transcribes each slice with Whisper for accurate per-scene lyrics.

    Args:
        project: The project to process
        boundaries: Scene time ranges from the waveform editor
        snap_to_beats: Snap boundaries to nearest beat time
        device: GPU device for Whisper (None = auto-detect)
    """
    config = project.config
    paths = project.paths

    audio_path = project.resolve_path(config.song.audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    beat_times = config.song.beat_times

    duration = config.song.duration_seconds or 0.0

    # Load word timestamps as fallback for lyrics population
    words: list[WordTimestamp] = []
    ts_path = paths.input_dir / "word_timestamps.json"
    if ts_path.exists():
        raw = json.loads(ts_path.read_text(encoding="utf-8"))
        words = [WordTimestamp(word=w["word"], start=w["start"], end=w["end"]) for w in raw]

    # Load raw lyrics text for BPM-based fallback
    lyrics_text = ""
    lyrics_path = project.resolve_path(config.song.lyrics_file) if config.song.lyrics_file else None
    if lyrics_path and lyrics_path.exists():
        lyrics_text = lyrics_path.read_text(encoding="utf-8")

    scenes: list[Scene] = []
    for i, b in enumerate(boundaries):
        t_start = b.time_start
        t_end = b.time_end

        if snap_to_beats and beat_times:
            t_start = _snap_to_beat(t_start, beat_times)
            t_end = _snap_to_beat(t_end, beat_times)

        # Pre-fill lyrics: try word timestamps, fall back to BPM-based estimation
        lyrics = _lyrics_from_words(words, t_start, t_end) if words else ""
        if not lyrics and lyrics_text:
            lyrics = _lyrics_for_scene_bpm(
                lyrics_text, t_start, t_end, duration,
                bpm=config.song.bpm,
            )

        scene_id = f"scene_{i + 1:03d}"
        scene = Scene(
            id=scene_id,
            order=i + 1,
            time_start=t_start,
            time_end=t_end,
            type=b.type,
            lyrics=lyrics,
            section=b.section,
        )
        scenes.append(scene)

    scene_list = SceneList(scenes=scenes)

    # Slice audio
    _slice_scenes(scene_list, audio_path, paths)

    # Transcribe each scene's audio slice for accurate lyrics
    if device is None:
        device = _detect_whisper_device()
    try:
        _transcribe_scene_slices(scene_list.scenes, paths, device=device)
    except Exception as exc:
        log.warning("Per-scene transcription failed, keeping word-timestamp lyrics: %s", exc)

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

    # Transcribe each scene's audio slice for accurate lyrics
    try:
        _transcribe_scene_slices(scene_list.scenes, paths, device=_detect_whisper_device())
    except Exception as exc:
        log.warning("Per-scene transcription failed, keeping LLM-assigned lyrics: %s", exc)

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


def _lyrics_from_words(
    words: list[WordTimestamp],
    time_start: float,
    time_end: float,
) -> str:
    """Extract lyrics from word timestamps within a time range.

    A word is included if its midpoint falls within [time_start, time_end].
    """
    scene_words = []
    for w in words:
        mid = (w.start + w.end) / 2
        if time_start <= mid <= time_end:
            scene_words.append(w.word)
    return " ".join(scene_words)


def _lyrics_for_scene_bpm(
    lyrics_text: str,
    scene_start: float,
    scene_end: float,
    song_duration: float,
    bpm: float | None = None,
) -> str:
    """Estimate which lyrics fall within a scene using BPM-based pacing.

    Used as fallback when Whisper word timestamps are unavailable or
    unreliable. Parses lyrics into lines (skipping section markers),
    estimates timing from BPM, and returns lines whose estimated time
    overlaps with the scene range.
    """
    lines: list[str] = []
    for line in lyrics_text.split("\n"):
        stripped = line.strip()
        if not stripped or re.match(r"^\s*\([^)]+\)\s*$", stripped):
            continue
        if stripped:
            lines.append(stripped)

    if not lines:
        return ""

    total_words = sum(len(line.split()) for line in lines)
    if total_words == 0:
        return ""

    effective_bpm = bpm if bpm and bpm > 0 else 120.0
    seconds_per_word = 60.0 / (effective_bpm * 2.0)
    line_pause = 60.0 / (effective_bpm * 2)

    # Estimate intro
    intro = min(2 * 4 * (60.0 / effective_bpm), song_duration * 0.1)

    # Compress if lyrics are too dense for the song
    singing_time = total_words * seconds_per_word + len(lines) * line_pause
    if singing_time > song_duration * 0.9:
        scale = (song_duration * 0.85) / singing_time
        seconds_per_word *= scale
        line_pause *= scale

    # Walk through lines, collect those that overlap with scene
    cursor = intro
    scene_lines: list[str] = []
    for line in lines:
        n_words = len(line.split())
        line_start = cursor
        line_end = cursor + n_words * seconds_per_word
        cursor = line_end + line_pause

        # Check overlap
        if line_end > scene_start and line_start < scene_end:
            scene_lines.append(line)

    return " / ".join(scene_lines)


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


def _detect_whisper_device() -> str:
    """Auto-detect the best device for Whisper (secondary GPU preferred)."""
    try:
        from musicvision.utils.gpu import detect_devices
        dm = detect_devices()
        return str(dm.secondary)
    except Exception:
        return "cpu"


def _transcribe_scene_slices(
    scenes: list[Scene],
    paths,
    device: str = "cuda:0",
) -> None:
    """Transcribe each scene's vocal audio slice for accurate per-scene lyrics.

    Short clips (2-10s) give Whisper much better accuracy than trying to
    use word timestamps from a full-song transcription, where timing drifts
    significantly on sung vocals.
    """
    # Check if any vocal segments exist
    has_vocal_slices = any(
        s.audio_segment_vocal and (paths.root / s.audio_segment_vocal).exists()
        for s in scenes
    )
    if not has_vocal_slices:
        # Fall back to full-mix segments
        has_mix_slices = any(
            s.audio_segment and (paths.root / s.audio_segment).exists()
            for s in scenes
        )
        if not has_mix_slices:
            log.warning("No audio segments available for per-scene transcription")
            return

    log.info("Transcribing %d scene slices with Whisper for accurate lyrics...", len(scenes))
    transcription_fn = transcribe  # import already at top

    # Load model once, transcribe all scenes
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline

    model_name = "openai/whisper-large-v3"
    torch_dtype = torch.float16 if "cuda" in device else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    for scene in scenes:
        # Prefer vocal slice for cleaner transcription
        audio_rel = scene.audio_segment_vocal or scene.audio_segment
        if not audio_rel:
            continue
        audio_file = paths.root / audio_rel
        if not audio_file.exists():
            continue

        try:
            result = pipe(
                str(audio_file),
                return_timestamps=False,
                generate_kwargs={"task": "transcribe"},
            )
            text = result.get("text", "").strip()
            if text:
                scene.lyrics = text
                log.debug("Scene %s lyrics: %s", scene.id, text[:80])
        except Exception as exc:
            log.warning("Failed to transcribe %s: %s", scene.id, exc)

    # Unload
    del pipe, model, processor
    if "cuda" in device:
        torch.cuda.empty_cache()
    log.info("Per-scene transcription complete")


def _snap_to_beat(t: float, beat_times: list[float], tolerance: float = 0.15) -> float:
    """Snap a time to the nearest beat within tolerance."""
    if not beat_times:
        return t
    closest = min(beat_times, key=lambda bt: abs(bt - t))
    if abs(closest - t) <= tolerance:
        return closest
    return t


def _approximate_word_timestamps(
    lyrics_text: str,
    duration: float,
    bpm: float | None = None,
) -> list[WordTimestamp]:
    """
    Create approximate word timestamps when Whisper data is unavailable.

    Uses BPM to estimate singing pace. Lyrics line breaks are treated as
    phrase boundaries with natural pauses. Section markers like (Verse 1)
    are skipped. Instrumental intro/outro time is estimated from BPM.

    This doesn't need to be frame-accurate — it's used to assign
    approximate lyrics to scenes for image/video prompt generation.
    """
    # Parse lines, skip section markers and blank lines
    lines: list[list[str]] = []
    for line in lyrics_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Skip section markers like (Verse 1), (Chorus), etc.
        if re.match(r"^\s*\([^)]+\)\s*$", stripped):
            continue
        words = stripped.split()
        if words:
            lines.append(words)

    if not lines:
        return []

    total_words = sum(len(line) for line in lines)
    if total_words == 0:
        return []

    # Estimate singing pace from BPM
    # Typical pop/rock: ~2 words per beat. Slower ballads: ~1.5, faster rap: ~4+
    words_per_beat = 2.0
    effective_bpm = bpm if bpm and bpm > 0 else 120.0
    seconds_per_word = 60.0 / (effective_bpm * words_per_beat)

    # Estimate total singing time vs. song duration
    # Assume ~10-15% of song is intro/outro/instrumental gaps
    singing_duration = total_words * seconds_per_word
    if singing_duration > duration * 0.9:
        # Lyrics are denser than expected — compress to fit
        seconds_per_word = (duration * 0.85) / total_words

    # Estimate intro (instrumental before vocals start) — ~10% or 2 bars
    bars_intro = 2
    beats_per_bar = 4
    intro_seconds = min(
        bars_intro * beats_per_bar * (60.0 / effective_bpm),
        duration * 0.1,
    )

    # Pause between lines (half a beat)
    line_pause = 60.0 / (effective_bpm * 2)

    result: list[WordTimestamp] = []
    cursor = intro_seconds

    for line_words in lines:
        for word in line_words:
            start = cursor
            end = cursor + seconds_per_word
            if end > duration:
                end = duration
            result.append(WordTimestamp(word=word, start=start, end=end))
            cursor = end
        # Pause between lines
        cursor += line_pause

    return result
