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
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from musicvision.models import SeparationMethod

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
    Vocal/instrumental separation using MelBandRoformer via the audio-separator library.

    The model auto-downloads on first use (~500 MB).
    Primary use: cleaner Whisper input + instrumental scene detection.

    Install: pip install "audio-separator[gpu]"

    Default model: MelBandRoformer.ckpt  (best quality, PyTorch)
    Fallback model: Kim_Vocal_2.onnx     (ONNX, slightly faster, same architecture)
    """

    DEFAULT_MODEL = "MelBandRoformer.ckpt"

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cuda:1",
        model_cache_dir: str = "/tmp/audio-separator-models",
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.model_cache_dir = model_cache_dir
        self._separator = None

    def load(self) -> None:
        """Load separation model. Downloads weights on first use (~500 MB)."""
        try:
            from audio_separator.separator import Separator
        except ImportError as e:
            raise RuntimeError(
                "audio-separator is not installed. "
                "Run: pip install 'audio-separator[gpu]'"
            ) from e

        log.info("Loading VocalSeparator: %s on %s", self.model_name, self.device)
        self._separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=self.model_cache_dir,
            output_format="WAV",
        )
        self._separator.load_model(self.model_name)
        log.info("VocalSeparator ready")

    def separate(
        self,
        audio_path: Path,
        output_vocal_path: Path,
        output_instrumental_path: Path | None = None,
    ) -> Path:
        """
        Separate vocals from the full mix.

        Args:
            audio_path: Input audio file (full mix).
            output_vocal_path: Destination for the isolated vocal stem.
            output_instrumental_path: Destination for the instrumental stem (optional).

        Returns:
            Path to the vocal output file.
        """
        if self._separator is None:
            raise RuntimeError("Call load() before separate()")

        import shutil
        import tempfile

        output_vocal_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            self._separator.output_dir = tmp
            output_files = self._separator.separate(str(audio_path))

            # Find stems by keyword in filename (model-agnostic)
            vocal_src: Path | None = None
            instr_src: Path | None = None
            for fname in output_files:
                lower = fname.lower()
                p = Path(tmp) / fname
                if not p.exists():
                    # audio-separator may return full paths in some versions
                    p = Path(fname)
                if "vocal" in lower and "no_vocal" not in lower and "instrumental" not in lower:
                    vocal_src = p
                elif "instrumental" in lower or "no_vocal" in lower or "other" in lower:
                    instr_src = p

            if vocal_src is None:
                raise RuntimeError(
                    f"VocalSeparator produced no vocal stem. Got: {output_files}. "
                    f"Try model_name='Kim_Vocal_2.onnx' as fallback."
                )

            shutil.move(str(vocal_src), str(output_vocal_path))
            log.info("Vocal stem → %s (%.1f MB)", output_vocal_path.name,
                     output_vocal_path.stat().st_size / 1e6)

            if output_instrumental_path is not None and instr_src is not None:
                output_instrumental_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(instr_src), str(output_instrumental_path))
                log.info("Instrumental stem → %s", output_instrumental_path.name)

        return output_vocal_path

    def unload(self) -> None:
        """Free model from memory."""
        if self._separator is not None:
            del self._separator
            self._separator = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("VocalSeparator unloaded")


class DemucsSeparator:
    """
    Vocal separation using Demucs (htdemucs / mdx_extra variants).

    Adapted from StemForge's DemucsPipeline with the same GPU-fallback logic and
    channel-normalisation conventions.  Often superior to MelBandRoFormer on
    AI-generated (synthetic) music because Demucs was trained on a wider palette
    of production styles.

    Models (listed best→slowest):
      htdemucs     — Hybrid Transformer, best overall (default)
      htdemucs_ft  — Fine-tuned on pop/rock, tighter transients
      mdx_extra    — MDX STFT-based, excellent vocal isolation
                     (auto-falls back to CPU on CUBLAS errors)
      mdx_extra_q  — Quantised MDX, cleanest output; requires diffq package

    Install:  pip install "demucs @ git+https://github.com/facebookresearch/demucs.git@v4.0.1"
    """

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str = "cuda:1",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._actual_device: str = "cpu"

    def load(self) -> None:
        """Load model weights. Downloads on first use (~300 MB per model)."""
        import torch

        try:
            from demucs.pretrained import get_model
        except ImportError as e:
            raise RuntimeError(
                "demucs is not installed. "
                "Run: pip install "
                "'demucs @ git+https://github.com/facebookresearch/demucs.git@v4.0.1'"
            ) from e

        log.info(
            "Loading Demucs model '%s' (may download weights on first run)",
            self.model_name,
        )
        try:
            self._model = get_model(self.model_name)
        except KeyError as e:
            raise RuntimeError(
                f"Unknown Demucs model '{self.model_name}'. "
                f"Valid: htdemucs, htdemucs_ft, mdx_extra, mdx_extra_q."
            ) from e

        self._actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        # Honour specific CUDA device index when on multi-GPU
        if self._actual_device == "cuda" and self.device.startswith("cuda:"):
            self._actual_device = self.device
        self._model = self._model.to(self._actual_device)
        self._model.eval()
        log.info(
            "DemucsSeparator ready: '%s' on %s (samplerate=%d, sources=%s)",
            self.model_name,
            self._actual_device,
            self._model.samplerate,
            self._model.sources,
        )

    def separate(
        self,
        audio_path: Path,
        output_vocal_path: Path,
        output_instrumental_path: Path | None = None,
    ) -> Path:
        """
        Separate vocals from the full mix using Demucs.

        The instrumental output is the sum of drums + bass + other stems,
        clipped to [-1, 1].  Both outputs are written at the model's native
        sample rate (44 100 Hz).

        Args:
            audio_path: Input audio file (full mix).
            output_vocal_path: Destination for isolated vocal stem.
            output_instrumental_path: Destination for instrumental stem (optional).

        Returns:
            Path to the vocal output file.
        """
        if self._model is None:
            raise RuntimeError("Call load() before separate()")

        import numpy as np
        import soundfile as sf
        import torch
        from demucs.apply import apply_model
        from demucs.audio import convert_audio

        # ------------------------------------------------------------------
        # 1. Load audio as float32 (channels, samples)
        # ------------------------------------------------------------------
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = data.T  # (channels, samples)

        # ------------------------------------------------------------------
        # 2. Channel normalisation (mirrors StemForge's _preprocess exactly)
        #    demucs.audio.convert_audio asserts src_channels in {1, model.audio_channels}
        #    so we guarantee stereo before handing off.
        # ------------------------------------------------------------------
        n_ch = waveform.shape[0]
        if n_ch == 1:
            waveform = np.repeat(waveform, 2, axis=0)          # mono → stereo
        elif n_ch > 2:
            waveform = waveform.mean(axis=0, keepdims=True)    # N→mono
            waveform = np.repeat(waveform, 2, axis=0)          # mono → stereo
        # n_ch == 2: pass through unchanged

        # ------------------------------------------------------------------
        # 3. Build batch tensor and resample to model's native rate
        # ------------------------------------------------------------------
        mix = torch.from_numpy(waveform).unsqueeze(0).float()  # (1, 2, T)
        mix = convert_audio(mix, sr, self._model.samplerate, self._model.audio_channels)

        # ------------------------------------------------------------------
        # 4. Inference with CUBLAS fallback for MDX models (from StemForge)
        # ------------------------------------------------------------------
        mix = mix.contiguous()
        is_mdx = self.model_name.startswith("mdx")
        apply_kwargs: dict = dict(progress=False, num_workers=0)
        if is_mdx and hasattr(self._model, "segment"):
            apply_kwargs["segment"] = self._model.segment

        def _apply(wav):
            with torch.no_grad():
                return apply_model(self._model, wav, **apply_kwargs)

        try:
            sources = _apply(mix.to(self._actual_device))
        except RuntimeError as exc:
            is_cuda_err = self._actual_device != "cpu" and (
                "CUBLAS" in str(exc) or "CUDA error" in str(exc)
            )
            if not is_cuda_err:
                raise
            log.warning(
                "Demucs GPU inference failed (%s). "
                "Retrying on CPU — htdemucs or htdemucs_ft avoid this error.",
                type(exc).__name__,
            )
            self._model.cpu()
            try:
                sources = _apply(mix.cpu())
            finally:
                self._model.to(self._actual_device)

        # ------------------------------------------------------------------
        # 5. Extract and save stems
        # ------------------------------------------------------------------
        # sources: (1, n_stems, channels, T) → drop batch dim → (n_stems, channels, T)
        sources = sources[0]

        if torch.isnan(sources).any() or torch.isinf(sources).any():
            raise RuntimeError(
                "Demucs produced NaN/Inf values — check input audio (clipping, silence)."
            )

        stem_names: list[str] = self._model.sources  # ['drums', 'bass', 'other', 'vocals']
        vocal_idx = stem_names.index("vocals")
        vocal_np = sources[vocal_idx].cpu().numpy().clip(-1.0, 1.0)  # (channels, T)

        output_vocal_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_vocal_path), vocal_np.T, self._model.samplerate)
        log.info(
            "Vocal stem → %s (%.1f MB)",
            output_vocal_path.name,
            output_vocal_path.stat().st_size / 1e6,
        )

        if output_instrumental_path is not None:
            non_vocal = torch.stack([
                sources[i] for i, name in enumerate(stem_names) if name != "vocals"
            ])
            instr_np = non_vocal.sum(dim=0).cpu().numpy().clip(-1.0, 1.0)  # (channels, T)
            output_instrumental_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_instrumental_path), instr_np.T, self._model.samplerate)
            log.info("Instrumental stem → %s", output_instrumental_path.name)

        return output_vocal_path

    def unload(self) -> None:
        """Evict model from GPU memory."""
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("DemucsSeparator unloaded")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_separator(
    method: "SeparationMethod",
    *,
    device: str = "cuda:1",
    roformer_model: str = "MelBandRoformer.ckpt",
    demucs_model: str = "htdemucs",
) -> "VocalSeparator | DemucsSeparator":
    """
    Create and return the appropriate separator for *method*.

    Args:
        method: SeparationMethod.ROFORMER or SeparationMethod.DEMUCS
        device: CUDA device string (e.g. "cuda:0", "cuda:1", "cpu")
        roformer_model: audio-separator model filename for ROFORMER method
        demucs_model: Demucs variant name for DEMUCS method
    """
    from musicvision.models import SeparationMethod as SM

    if method == SM.ROFORMER:
        return VocalSeparator(model_name=roformer_model, device=device)
    if method == SM.DEMUCS:
        return DemucsSeparator(model_name=demucs_model, device=device)
    raise ValueError(f"Unknown separation method: {method}")
