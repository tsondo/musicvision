"""
Stage 1 orchestrator: Intake & Segmentation.

Coordinates the full intake pipeline:
  1. Import audio → get duration
  2. Detect BPM + beat times
  3. Transcribe with Whisper (or load provided lyrics)
  4. Segment into scenes (LLM or rule-based)
  5. Slice audio into per-scene segments

This is what the API endpoint and CLI call.
"""

from __future__ import annotations

import logging
from pathlib import Path

from musicvision.intake.audio_analysis import detect_bpm, get_beat_times
from musicvision.intake.segmentation import segment_scenes, segment_scenes_simple
from musicvision.intake.transcription import (
    TranscriptionResult,
    WordTimestamp,
    align_lyrics_with_timestamps,
    load_lyrics_file,
    transcribe,
)
from musicvision.models import SceneList
from musicvision.project import ProjectService
from musicvision.utils.audio import get_duration, slice_audio

log = logging.getLogger(__name__)


def run_intake(
    project: ProjectService,
    use_llm_segmentation: bool = True,
    whisper_device: str = "cuda:1",
    skip_transcription: bool = False,
) -> SceneList:
    """
    Run the full Stage 1 pipeline.

    Args:
        project: The project to process
        use_llm_segmentation: Use Claude API for segmentation (vs rule-based)
        whisper_device: Device for Whisper inference
        skip_transcription: Skip Whisper if lyrics are already provided

    Returns:
        SceneList (also saved to project)
    """
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

    # --- Beat times (for boundary snapping) ---
    beat_times = get_beat_times(audio_path)
    log.info("Detected %d beats", len(beat_times))

    # --- Transcription / Lyrics ---
    words: list[WordTimestamp] = []

    lyrics_path = (
        project.resolve_path(config.song.lyrics_file)
        if config.song.lyrics_file
        else None
    )
    has_lyrics = lyrics_path and lyrics_path.exists()

    if skip_transcription and has_lyrics:
        # User provided lyrics, skip Whisper entirely.
        # We'll do simple segmentation based on line breaks.
        log.info("Using provided lyrics, skipping transcription")
        lyrics_text = load_lyrics_file(lyrics_path)
        # Create approximate word timestamps from line structure
        # (This is rough — for precise timing, run Whisper and align)
        words = _approximate_word_timestamps(lyrics_text, duration)

    elif has_lyrics:
        # User provided lyrics but we still run Whisper for timestamps,
        # then align the user's lyrics with Whisper's timing.
        log.info("Transcribing for timestamps, will align with provided lyrics")
        transcription = transcribe(audio_path, device=whisper_device)
        lyrics_text = load_lyrics_file(lyrics_path)
        words = align_lyrics_with_timestamps(lyrics_text, transcription)
        log.info("Aligned %d words from provided lyrics", len(words))

    else:
        # No lyrics provided — Whisper does everything
        log.info("No lyrics file — transcribing with Whisper")
        transcription = transcribe(audio_path, device=whisper_device)
        words = transcription.words

        # Save transcription as lyrics file
        lyrics_out = paths.input_dir / "lyrics_whisper.txt"
        lyrics_out.write_text(transcription.text, encoding="utf-8")
        config.song.lyrics_file = f"input/lyrics_whisper.txt"
        log.info("Saved Whisper transcription to %s", lyrics_out.name)

    # --- Segmentation ---
    acestep_caption = None
    acestep_lyrics = None
    if has_acestep:
        acestep_caption = config.song.acestep.caption
        acestep_lyrics = config.song.acestep.lyrics
        log.info("Passing AceStep caption + section-marked lyrics to segmenter")

    if use_llm_segmentation and words:
        log.info("Segmenting with Claude API...")
        scene_list = segment_scenes(
            lyrics_with_timestamps=words,
            song_duration=duration,
            bpm=bpm,
            beat_times=beat_times,
            acestep_caption=acestep_caption,
            acestep_lyrics=acestep_lyrics,
        )
    else:
        log.info("Segmenting with rule-based splitter...")
        scene_list = segment_scenes_simple(
            lyrics_with_timestamps=words,
            song_duration=duration,
        )

    # --- Slice audio into per-scene segments ---
    log.info("Slicing audio into %d segments...", len(scene_list.scenes))
    for scene in scene_list.scenes:
        segment_path = paths.segment_path(scene.id)
        slice_audio(audio_path, segment_path, scene.time_start, scene.time_end)
        scene.audio_segment = str(segment_path.relative_to(paths.root))

    # --- Save ---
    project.config = config
    project.scenes = scene_list
    project.save_all()

    log.info(
        "Intake complete: %d scenes (%.1fs total), saved to %s",
        len(scene_list.scenes),
        scene_list.total_duration,
        paths.scenes_file,
    )

    return scene_list


def _approximate_word_timestamps(lyrics_text: str, duration: float) -> list[WordTimestamp]:
    """
    Create rough word timestamps when we have lyrics but no Whisper.

    Distributes words evenly across the song duration.
    Very approximate — only useful for rule-based segmentation.
    """
    words_raw = lyrics_text.split()
    if not words_raw:
        return []

    # Estimate ~0.3s per word, spread across duration
    word_duration = min(0.3, duration / len(words_raw))
    result: list[WordTimestamp] = []

    for i, word in enumerate(words_raw):
        start = i * word_duration
        end = start + word_duration
        result.append(WordTimestamp(word=word, start=start, end=min(end, duration)))

    return result
