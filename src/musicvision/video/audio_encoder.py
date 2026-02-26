"""
HumoAudioEncoder — Whisper-based audio feature extraction for HuMo TIA mode.

Extracts multi-band temporal features from audio that are used by the
AudioCrossAttentionWrapper in WanModel to synchronize video generation
with music.

Pipeline
--------
  1. Load audio → mono float32 @ 16 kHz
  2. Extract Whisper encoder hidden states (all 33 layers of whisper-large-v3)
  3. Average hidden states into 5 frequency bands
  4. Interpolate band features from Whisper's 50 fps to video fps (25 fps)
  5. Extract a sliding window of 8 consecutive frames per latent frame
  6. Optionally prepend one window for the reference (conditioning) frame

Output shape: [1, total_frames, 8, 5, 1280]
  total_frames = num_latent_frames + 1   (when include_ref_frame=True)

Reference: bytedance-research/HuMo audio conditioning pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Whisper large-v3 hidden-state count: embedding + 32 transformer layers = 33
_WHISPER_NUM_HIDDEN = 33
# Whisper encoder output rate: 1 frame per 20 ms → 50 fps
_WHISPER_FPS = 50
# Whisper target sample rate
_WHISPER_SR  = 16_000
# Band boundaries over the 33 hidden-state layers (matching upstream HuMo)
# Bands 0-3: uniform 8-layer groups; Band 4: final layer only
_BAND_BOUNDARIES = [(0, 8), (8, 16), (16, 24), (24, 32), (32, 33)]
_NUM_BANDS       = len(_BAND_BOUNDARIES)   # 5
# Whisper large-v3 hidden dim
_WHISPER_DIM = 1280


class HumoAudioEncoder:
    """
    Audio feature extractor for HuMo TIA (Text-Image-Audio) mode.

    Wraps a pre-loaded Whisper encoder model and converts raw audio into
    per-frame, per-band temporal feature windows that the DiT can attend to.

    Usage::

        from transformers import WhisperModel
        whisper = WhisperModel.from_pretrained("openai/whisper-large-v3")
        enc = HumoAudioEncoder(whisper.encoder, device=torch.device("cuda"))
        features = enc.encode(Path("segment.wav"), num_latent_frames=25)
        # features: [1, 26, 8, 5, 1280]  (25 + 1 ref frame)
    """

    def __init__(
        self,
        whisper_model,
        device,
        target_fps: int = 25,
        window_size: int = 8,
    ) -> None:
        """
        Args:
            whisper_model: Loaded Whisper encoder (transformers WhisperEncoder or
                           the encoder sub-module of WhisperModel).
            device:        torch.device (or str) for computation.
            target_fps:    Video frame rate — 25 for HuMo.
            window_size:   Number of Whisper feature frames per latent window.
                           8 frames at 50 fps = 160 ms context per latent frame.
        """
        import torch  # lazy

        self.whisper_model = whisper_model
        self.device        = torch.device(device) if isinstance(device, str) else device
        self.target_fps    = target_fps
        self.window_size   = window_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        audio_path: Path,
        num_latent_frames: int,
        include_ref_frame: bool = True,
    ) -> "torch.Tensor":
        """
        Extract audio conditioning features for a single video clip.

        Args:
            audio_path:        Path to the audio segment (WAV, any sample rate).
            num_latent_frames: Number of latent frames the DiT will generate
                               (e.g. 25 for a ~3.88 s clip at 97 video frames).
            include_ref_frame: If True, prepend one extra window for the
                               reference-image conditioning frame.

        Returns:
            Float tensor of shape [1, total_frames, 8, 5, 1280] on self.device,
            where total_frames = num_latent_frames + (1 if include_ref_frame else 0).
        """
        wav, sr = self._load_audio(audio_path)
        duration = len(wav) / sr

        log.debug(
            "Audio loaded: %.2f s, %d samples @ %d Hz", duration, len(wav), sr
        )

        hidden_states = self._extract_whisper_features(wav)
        # hidden_states: [1, whisper_seq_len, 33, 1280]

        bands = self._split_bands(hidden_states)
        # bands: [1, whisper_seq_len, 5, 1280]

        bands_video = self._interpolate_to_fps(bands, duration)
        # bands_video: [1, video_frames, 5, 1280]

        # Build windows for each latent frame
        windows = self._extract_windows(bands_video, num_latent_frames)
        # windows: [1, num_latent_frames, 8, 5, 1280]

        if include_ref_frame:
            # Reference frame window uses the same features as latent frame 0
            ref_window = windows[:, :1]   # [1, 1, 8, 5, 1280]
            windows    = _concat_tensors(ref_window, windows, dim=1)
            # → [1, num_latent_frames + 1, 8, 5, 1280]

        log.debug(
            "Audio features shape: %s (include_ref=%s)", list(windows.shape), include_ref_frame
        )
        return windows

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def _load_audio(self, path: Path) -> tuple[np.ndarray, int]:
        """
        Load audio file and return mono float32 @ 16 kHz.

        Args:
            path: Path to audio file (WAV, FLAC, MP3, etc.).

        Returns:
            (wav_mono_float32, sample_rate=16000)
        """
        import soundfile as sf

        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        # wav: [num_samples, num_channels]

        # Convert to mono by averaging across channels
        if wav.shape[1] > 1:
            wav = wav.mean(axis=1)
        else:
            wav = wav[:, 0]
        # wav: [num_samples]

        if sr != _WHISPER_SR:
            log.debug("Resampling audio %d Hz → %d Hz", sr, _WHISPER_SR)
            import librosa  # lazy — heavy dep
            wav = librosa.resample(wav, orig_sr=sr, target_sr=_WHISPER_SR)
            sr  = _WHISPER_SR

        return wav.astype(np.float32), sr

    # ------------------------------------------------------------------
    # Whisper feature extraction
    # ------------------------------------------------------------------

    def _extract_whisper_features(self, wav: np.ndarray) -> "torch.Tensor":
        """
        Run the Whisper encoder on raw audio and return all hidden states.

        Args:
            wav: Mono float32 waveform @ 16 kHz.

        Returns:
            Tensor [1, whisper_seq_len, 33, 1280]:
              - dim 1: temporal sequence (~50 fps)
              - dim 2: layer index (0 = embedding output, 1–32 = transformer layers)
              - dim 3: hidden dimension (1280 for whisper-large-v3)
        """
        import torch
        from transformers import WhisperFeatureExtractor

        # Try to resolve the feature extractor from the whisper model config
        try:
            fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
        except Exception:
            log.warning(
                "Could not load WhisperFeatureExtractor from HuggingFace; "
                "falling back to default params."
            )
            fe = WhisperFeatureExtractor()

        inputs = fe(
            wav,
            sampling_rate=_WHISPER_SR,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(
            device=self.device, dtype=torch.float32
        )
        # input_features: [1, n_mels=128, time_frames=3000]

        with torch.no_grad():
            enc_out = self.whisper_model(
                input_features,
                output_hidden_states=True,
            )

        # enc_out.hidden_states is a tuple of (embedding_output, layer_1, …, layer_32)
        # Length = 33 for whisper-large-v3
        all_hidden = enc_out.hidden_states
        if len(all_hidden) != _WHISPER_NUM_HIDDEN:
            log.warning(
                "Expected %d Whisper hidden states, got %d — proceeding anyway.",
                _WHISPER_NUM_HIDDEN,
                len(all_hidden),
            )

        # Stack: each element is [1, seq_len, 1280]
        stacked = torch.stack(
            [all_hidden[i] for i in range(len(all_hidden))], dim=2
        )
        # stacked: [1, seq_len, num_layers, 1280]
        return stacked

    # ------------------------------------------------------------------
    # Band splitting
    # ------------------------------------------------------------------

    def _split_bands(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
        """
        Average Whisper hidden-state layers into 5 frequency-analogue bands.

        The layer grouping approximates the spectral hierarchy of Whisper:
          - Band 0 (layers  0-6):  low-level acoustic / spectral features
          - Band 1 (layers  7-13): phoneme-level patterns
          - Band 2 (layers 14-19): syllable / word boundaries
          - Band 3 (layers 20-26): prosody / rhythm
          - Band 4 (layers 27-32): high-level semantic / musical structure

        Args:
            hidden_states: [1, seq_len, num_layers, 1280]

        Returns:
            [1, seq_len, 5, 1280]
        """
        import torch

        bands = []
        for start, end in _BAND_BOUNDARIES:
            # Clamp indices to available layer count
            end_clamped = min(end, hidden_states.shape[2])
            start_clamped = min(start, end_clamped)
            band = hidden_states[:, :, start_clamped:end_clamped, :].mean(dim=2)
            # band: [1, seq_len, 1280]
            bands.append(band)

        return torch.stack(bands, dim=2)
        # [1, seq_len, 5, 1280]

    # ------------------------------------------------------------------
    # FPS interpolation
    # ------------------------------------------------------------------

    def _interpolate_to_fps(
        self, bands: "torch.Tensor", duration: float
    ) -> "torch.Tensor":
        """
        Resample band features from Whisper's 50 fps to the target video fps.

        Args:
            bands:    [1, whisper_seq_len, 5, 1280]
            duration: Audio duration in seconds.

        Returns:
            [1, target_video_frames, 5, 1280]
        """
        import torch
        import torch.nn.functional as F

        target_frames = max(1, int(duration * self.target_fps))

        B, seq_len, num_bands, dim = bands.shape  # 1, seq, 5, 1280

        # Reshape to [B, channels, seq_len] for F.interpolate
        # Treat (num_bands * dim) as a flat channel axis
        x = bands.permute(0, 2, 3, 1)                # [1, 5, 1280, seq_len]
        x = x.reshape(B, num_bands * dim, seq_len)    # [1, 6400, seq_len]

        x = F.interpolate(
            x.float(),
            size=target_frames,
            mode="linear",
            align_corners=False,
        ).to(bands.dtype)
        # x: [1, 6400, target_frames]

        x = x.reshape(B, num_bands, dim, target_frames)    # [1, 5, 1280, frames]
        x = x.permute(0, 3, 1, 2)                          # [1, frames, 5, 1280]
        return x

    # ------------------------------------------------------------------
    # Window extraction
    # ------------------------------------------------------------------

    def _extract_windows(
        self, bands: "torch.Tensor", num_latent_frames: int
    ) -> "torch.Tensor":
        """
        Extract a fixed-size temporal window of audio features per latent frame.

        The video VAE has a temporal stride of 4, so latent frame i corresponds
        to video frame i * 4.  We extract a window of self.window_size
        consecutive feature frames centred on that position.

        Edge behaviour: replicate (clamp) the boundary frames rather than
        zero-pad, so that the model always receives meaningful audio context.

        Args:
            bands:             [1, video_frames, 5, 1280]
            num_latent_frames: Number of output windows to produce.

        Returns:
            [1, num_latent_frames, window_size, 5, 1280]
        """
        import torch

        B, video_frames, num_bands, dim = bands.shape
        ws = self.window_size
        half = ws // 2

        # Latent-to-video frame stride (matches VAE temporal compression ×4)
        latent_stride = 4

        windows = []
        for lat_idx in range(num_latent_frames):
            centre = lat_idx * latent_stride

            # Frame indices for this window (before boundary clamping)
            frame_indices = [
                max(0, min(video_frames - 1, centre - half + w))
                for w in range(ws)
            ]

            # Gather frames: each [1, 5, 1280]
            window_frames = torch.stack(
                [bands[:, fi] for fi in frame_indices], dim=1
            )
            # window_frames: [1, ws, 5, 1280]
            windows.append(window_frames)

        result = torch.stack(windows, dim=1)
        # result: [1, num_latent_frames, ws, 5, 1280]
        return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _concat_tensors(a: "torch.Tensor", b: "torch.Tensor", dim: int) -> "torch.Tensor":
    """torch.cat wrapper kept out of the class body to avoid import at module level."""
    import torch
    return torch.cat([a, b], dim=dim)
