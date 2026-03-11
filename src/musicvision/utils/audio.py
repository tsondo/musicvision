"""
Audio utilities — ffmpeg wrappers for slicing, muxing, and conversion.

All audio manipulation goes through ffmpeg subprocess calls.
No ffmpeg Python bindings — they add complexity and break on WSL more often than the CLI.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _check_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found in PATH. Install with: sudo apt install ffmpeg")
    return path


def detect_silences(
    audio_path: Path,
    min_silence_duration: float = 0.15,
    silence_threshold_db: float = -35.0,
) -> list[tuple[float, float]]:
    """
    Detect silence intervals in an audio file using ffmpeg silencedetect.

    Args:
        audio_path: Path to audio file (any format ffmpeg supports).
        min_silence_duration: Minimum silence length in seconds to report.
        silence_threshold_db: Volume threshold below which audio is considered silent.

    Returns:
        List of (start, end) tuples in seconds. Sorted by start time.
    """
    _check_ffmpeg()

    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-af", f"silencedetect=noise={silence_threshold_db}dB:d={min_silence_duration}",
        "-f", "null", "-",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    # silencedetect writes to stderr
    stderr = result.stderr

    starts: list[float] = []
    ends: list[float] = []
    for line in stderr.splitlines():
        m_start = re.search(r"silence_start:\s*([\d.]+)", line)
        if m_start:
            starts.append(float(m_start.group(1)))
            continue
        m_end = re.search(r"silence_end:\s*([\d.]+)", line)
        if m_end:
            ends.append(float(m_end.group(1)))

    # Pair starts and ends. If a silence is still active at EOF, ffmpeg may
    # emit a start without a matching end — cap it at the file duration.
    silences: list[tuple[float, float]] = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else s + min_silence_duration
        silences.append((s, e))

    log.debug("Detected %d silence intervals in %s", len(silences), audio_path.name)
    return silences


def get_duration(audio_path: Path) -> float:
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def slice_audio(
    source: Path,
    output: Path,
    start_seconds: float,
    end_seconds: float,
    sample_rate: int | None = None,
) -> Path:
    """
    Extract a segment from an audio file.

    Args:
        source: Input audio file
        output: Output WAV path
        start_seconds: Start time
        end_seconds: End time
        sample_rate: Optional resample rate (None = keep original)

    Returns:
        Path to the output file
    """
    _check_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)

    duration = end_seconds - start_seconds
    # -ss before -i uses fast input seek; output codec pcm_s16le (WAV) is
    # sample-accurate because PCM has no codec frame boundaries.
    # WARNING: if source is ever MP3/AAC/Opus, switch to -af atrim to avoid
    # sub-frame drift at the cut point.  WAV → WAV is always exact.
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_seconds:.6f}",
        "-i", str(source),
        "-t", f"{duration:.6f}",
        "-vn",  # no video
    ]

    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])

    cmd.extend(["-c:a", "pcm_s16le", str(output)])

    log.debug("Slicing audio: %s → %s (%.2fs – %.2fs)", source.name, output.name, start_seconds, end_seconds)
    subprocess.run(cmd, capture_output=True, check=True)
    return output


def mux_video_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    audio_offset: float = 0.0,
) -> Path:
    """
    Combine a video file (no audio) with an audio track.

    HuMo outputs video-only MP4. This muxes the original audio back.
    """
    _check_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
    ]

    if audio_offset > 0:
        cmd.extend(["-itsoffset", f"{audio_offset:.6f}"])

    cmd.extend([
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ])

    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def concat_videos(clip_paths: list[Path], output_path: Path) -> Path:
    """
    Concatenate multiple video clips using ffmpeg's concat demuxer.

    All clips must have the same resolution, codec, and framerate.
    """
    _check_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write concat list file
    list_file = output_path.parent / "_concat_list.txt"
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    list_file.unlink(missing_ok=True)
    return output_path


def slice_subclip_audio(
    scene_audio: Path,
    scene_id: str,
    subclip_frames: list[int],
    fps: int,
    output_dir: Path,
) -> list[Path]:
    """
    Slice a scene's audio segment into sub-clip audio files.

    Timing is derived from frame counts to maintain frame-accuracy.
    Called AFTER compute_subclip_frames(), BEFORE video generation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    cursor_frames = 0

    for i, n_frames in enumerate(subclip_frames):
        start_sec = cursor_frames / fps
        end_sec = (cursor_frames + n_frames) / fps

        sub_audio = output_dir / f"{scene_id}_sub_{i:02d}.wav"
        slice_audio(scene_audio, sub_audio, start_sec, end_sec)
        paths.append(sub_audio)

        cursor_frames += n_frames

    return paths


def generate_silence(output: Path, duration: float, sample_rate: int = 16000) -> Path:
    """Generate a silent WAV file of the given duration."""
    _check_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl=mono",
        "-t", f"{duration:.6f}",
        "-c:a", "pcm_s16le",
        str(output),
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    log.debug("Generated silence: %s (%.2fs)", output.name, duration)
    return output


def build_mixed_audio(
    original_audio: Path,
    scenes: list,
    project_root: Path,
    output: Path,
) -> Path | None:
    """Build a mixed audio track with per-scene generated audio overlaid on the song.

    For each scene, the audio mode controls what happens:
      - ``song_only``: original song plays through (default, no processing needed)
      - ``generated_only``: song is silenced; generated audio plays alone
      - ``mix``: generated audio layered over ducked song

    If no scene uses generated audio, returns None (caller should use original).

    The mixing is done in a single ffmpeg filter_complex call with volume
    automation via ``if(between())`` expressions evaluated per-frame.

    Args:
        original_audio: Path to the full song file.
        scenes: List of Scene objects (must have audio mixing fields).
        project_root: Project root for resolving relative paths.
        output: Where to write the mixed WAV.

    Returns:
        Path to the mixed audio file, or None if no mixing needed.
    """
    from musicvision.models import SceneAudioMode

    # Collect scenes that need audio mixing
    mix_scenes = []
    for scene in scenes:
        if scene.audio_mode == SceneAudioMode.SONG_ONLY:
            continue
        if not scene.generated_audio:
            log.warning(
                "Scene %s has audio_mode=%s but no generated_audio — treating as song_only",
                scene.id, scene.audio_mode.value,
            )
            continue
        gen_path = Path(scene.generated_audio)
        if not gen_path.is_absolute():
            gen_path = project_root / gen_path
        if not gen_path.exists():
            log.warning(
                "Scene %s generated audio missing: %s — treating as song_only",
                scene.id, gen_path,
            )
            continue
        mix_scenes.append((scene, gen_path))

    if not mix_scenes:
        return None

    _check_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg filter_complex:
    # Input 0 = original song
    # Inputs 1..N = generated audio files (one per mix scene)
    #
    # Song volume automation: duck during mix scenes, silence during generated_only
    # Generated audio: volume + fade in/out, placed at correct time offset

    inputs = ["-i", str(original_audio)]
    for _, gen_path in mix_scenes:
        inputs.extend(["-i", str(gen_path)])

    # --- Song volume expression ---
    # Each scene contributes a volume factor: 1.0 outside, target_vol inside,
    # with smooth fades at boundaries.  Factors multiply for overlap safety.
    duck_terms = []
    for scene, _ in mix_scenes:
        t0 = scene.time_start
        t1 = scene.time_end
        target = 0.0 if scene.audio_mode == SceneAudioMode.GENERATED_ONLY else scene.song_duck_volume
        fi = max(scene.song_duck_fade_in, 0.001)
        fo = max(scene.song_duck_fade_out, 0.001)
        # Fade-in region: lerp 1→target over [t0-fi, t0]
        # Hold region: target over [t0, t1]
        # Fade-out region: lerp target→1 over [t1, t1+fo]
        # Outside: 1.0
        duck_terms.append(
            f"if(between(t,{t0:.4f},{t1:.4f}),{target:.4f},"
            f"if(between(t,{max(0, t0-fi):.4f},{t0:.4f}),"
            f"1-(1-{target:.4f})*(1-({t0:.4f}-t)/{fi:.4f}),"
            f"if(between(t,{t1:.4f},{t1+fo:.4f}),"
            f"{target:.4f}+(1-{target:.4f})*(t-{t1:.4f})/{fo:.4f},1)))"
        )

    if duck_terms:
        song_vol_expr = "*".join(duck_terms)
        song_filter = f"[0:a]volume='{song_vol_expr}':eval=frame[song]"
    else:
        song_filter = "[0:a]acopy[song]"

    # --- Generated audio overlay filters ---
    gen_filters = []
    gen_labels = []
    for i, (scene, _) in enumerate(mix_scenes):
        input_idx = i + 1
        vol = scene.generated_audio_volume
        fade_in = scene.audio_fade_in
        fade_out = scene.audio_fade_out
        duration = scene.time_end - scene.time_start

        # Apply volume + fades to generated audio
        af_parts = [f"volume={vol:.4f}"]
        if fade_in > 0:
            af_parts.append(f"afade=t=in:d={fade_in:.4f}")
        if fade_out > 0:
            af_parts.append(f"afade=t=out:st={max(0, duration - fade_out):.4f}:d={fade_out:.4f}")

        label = f"gen{i}"
        gen_filters.append(
            f"[{input_idx}:a]{','.join(af_parts)},"
            f"adelay={int(scene.time_start * 1000)}|{int(scene.time_start * 1000)},"
            f"apad=whole_dur=0[{label}]"
        )
        gen_labels.append(f"[{label}]")

    # --- Amerge: overlay all generated audio onto the ducked song ---
    filter_parts = [song_filter] + gen_filters

    if gen_labels:
        # amix all streams together
        all_labels = "[song]" + "".join(gen_labels)
        n_inputs = 1 + len(gen_labels)
        filter_parts.append(
            f"{all_labels}amix=inputs={n_inputs}:duration=first:dropout_transition=0[out]"
        )
    else:
        filter_parts.append("[song]acopy[out]")

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:a", "pcm_s16le",
        str(output),
    ]

    log.info("Building mixed audio with %d generated audio overlay(s)", len(mix_scenes))
    log.debug("ffmpeg filter_complex: %s", filter_complex)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ffmpeg mixed audio failed:\n%s", result.stderr)
        raise RuntimeError(f"ffmpeg mixed audio failed: {result.stderr[-500:]}")

    log.info("Mixed audio saved: %s", output)
    return output


def convert_to_wav(source: Path, output: Path, sample_rate: int = 16000) -> Path:
    """Convert any audio format to WAV (mono, 16-bit PCM). Used for Whisper input."""
    _check_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(output),
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    return output
