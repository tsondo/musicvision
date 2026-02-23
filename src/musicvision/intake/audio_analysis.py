"""
Audio analysis: BPM detection, beat times, vocal activity detection.

BPM detection uses librosa.
Vocal separation uses MelBandRoformer (Kim_Vocal_2) — deferred until
model weights are available locally.

Note on vocal separation for HuMo:
  HuMo TIA mode takes the FULL MIX as audio input (not isolated vocals).
  We separate vocals for:
    1. Whisper transcription (cleaner vocals → better word timestamps)
    2. Instrumental scene detection (no vocals = instrumental section)
    3. Future: per-scene scale_a adjustment based on vocal presence
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def detect_bpm(audio_path: Path) -> float:
    """Detect dominant BPM using librosa's beat tracker."""
    import librosa

    y, sr = librosa.load(str(audio_path), sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    log.info("Detected BPM: %.1f from %s", bpm, audio_path.name)
    return bpm


def get_beat_times(audio_path: Path) -> list[float]:
    """Get timestamps of detected beats in seconds. Useful for scene boundary snapping."""
    import librosa

    y, sr = librosa.load(str(audio_path), sr=None)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return [float(t) for t in beat_times]


def estimate_vocal_activity(
    vocal_audio_path: Path,
    frame_length_ms: float = 50.0,
    energy_threshold_db: float = -40.0,
) -> list[tuple[float, float]]:
    """
    Estimate time ranges where vocals are active (energy-based).

    Returns list of (start, end) tuples in seconds.
    Used to classify scenes as vocal vs instrumental.
    """
    import librosa

    y, sr = librosa.load(str(vocal_audio_path), sr=None)
    frame_length = int(sr * frame_length_ms / 1000)
    hop_length = frame_length // 2

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    active = rms_db > energy_threshold_db
    times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop_length)

    regions: list[tuple[float, float]] = []
    in_region = False
    start = 0.0

    for i, is_active in enumerate(active):
        if is_active and not in_region:
            start = float(times[i])
            in_region = True
        elif not is_active and in_region:
            regions.append((start, float(times[i])))
            in_region = False

    if in_region:
        regions.append((start, float(times[-1])))

    return regions


class VocalSeparator:
    """
    Vocal separation using MelBandRoformer (Kim_Vocal_2).

    Deferred — requires model weights on disk.
    Model: huggingface.co/Kijai/MelBandRoFormer_comfy/MelBandRoformer_fp16.safetensors
    """

    def __init__(self, model_path: str | None = None, device: str = "cuda:1"):
        self.model_path = model_path
        self.device = device
        self._model = None

    def load(self) -> None:
        raise NotImplementedError(
            "Vocal separation requires MelBandRoformer_fp16.safetensors. "
            "Download from huggingface.co/Kijai/MelBandRoFormer_comfy"
        )

    def separate(
        self,
        audio_path: Path,
        output_vocal_path: Path,
        output_instrumental_path: Path | None = None,
    ) -> Path:
        raise NotImplementedError("Vocal separation not yet implemented")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
