"""
HuMo inference wrapper for video generation.

Wraps the HuMo model (built on Wan2.1-T2V) for TIA mode generation.
Model loading and inference patterns adapted from kijai's ComfyUI-WanVideoWrapper.

Key reference files in ComfyUI-WanVideoWrapper:
  - nodes_model_loading.py  → WanVideoModelLoader (block swap, fp8 quantization)
  - nodes_sampler.py        → WanVideoSampler (denoising loop with HuMo conditioning)
  - nodes.py                → HuMoEmbeds (reference image + audio → conditioning tensors)

The ComfyUI wrapper uses comfy.model_management for memory management.
We replace that with our own DeviceMap-based approach.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path

import torch

from musicvision.models import HumoConfig, HumoModelSize
from musicvision.utils.audio import slice_audio
from musicvision.utils.gpu import DeviceMap, clear_vram
from musicvision.video.base import VideoEngine, VideoInput, VideoResult

log = logging.getLogger(__name__)

# HuMo constants
MAX_FRAMES = 97          # 97 frames @ 25fps = 3.88 seconds
FPS = 25
MAX_DURATION = MAX_FRAMES / FPS  # 3.88s

# HuggingFace model IDs
WAN_MODEL_IDS: dict[HumoModelSize, str] = {
    HumoModelSize.LARGE: "Wan-AI/Wan2.1-T2V-14B",
    HumoModelSize.SMALL: "Wan-AI/Wan2.1-T2V-1.3B",
}

HUMO_REPO = "bytedance-research/HuMo"
WHISPER_MODEL = "openai/whisper-large-v3"


class HumoEngine(VideoEngine):
    """
    HuMo video generation engine (TIA mode).

    Lifecycle:
        engine = HumoEngine(config, device_map)
        engine.load()
        result = engine.generate(input)
        engine.unload()

    Multi-GPU strategy (mirrors ComfyUI-WanVideoWrapper):
        - DiT (transformer blocks) → primary GPU (5080)
        - T5 text encoder → secondary GPU (3080 Ti), unloaded after encoding
        - Whisper encoder → secondary GPU, unloaded after encoding
        - VAE decoder → secondary GPU
        - Block swap: offload N transformer blocks to CPU, swap in during inference
          (from WanVideoBlockSwap node in ComfyUI)

    TIA mode conditioning:
        1. Encode text prompt via T5 → text embeddings
        2. Encode reference image → image condition latents (HuMoEmbeds)
        3. Encode audio via Whisper encoder → audio embeddings
        4. Run denoising with both audio and image conditioning
        5. Decode latents via VAE → frames
        6. Save as MP4 (video only, no audio — mux separately)
    """

    def __init__(self, config: HumoConfig, device_map: DeviceMap):
        self.config = config
        self.device_map = device_map
        self._dit = None
        self._t5_encoder = None
        self._t5_tokenizer = None
        self._vae = None
        self._whisper_encoder = None
        self._whisper_processor = None
        self._image_processor = None
        self._scheduler = None

    @property
    def is_loaded(self) -> bool:
        return self._dit is not None

    def load(self) -> None:
        """
        Load HuMo model components with multi-GPU split.

        Components:
        1. DiT with CPU offload on primary GPU
        2. T5 text encoder on secondary GPU
        3. Whisper encoder on secondary GPU
        4. VAE on secondary GPU
        """
        from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            CLIPImageProcessor,
            T5EncoderModel,
            T5Tokenizer,
        )

        wan_model_id = WAN_MODEL_IDS[self.config.model_size]
        primary = self.device_map.dit_device
        secondary = self.device_map.encoder_device

        # 1. Load scheduler
        log.info("Loading scheduler from %s", wan_model_id)
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            wan_model_id, subfolder="scheduler"
        )

        # 2. Load T5 text encoder on secondary GPU
        log.info("Loading T5 text encoder on %s", secondary)
        self._t5_tokenizer = T5Tokenizer.from_pretrained(
            wan_model_id, subfolder="tokenizer"
        )
        self._t5_encoder = T5EncoderModel.from_pretrained(
            wan_model_id,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        ).to(secondary)

        # 3. Load VAE on secondary GPU
        log.info("Loading VAE on %s", secondary)
        self._vae = AutoencoderKLWan.from_pretrained(
            wan_model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(secondary)

        # 4. Load Whisper encoder on secondary GPU
        log.info("Loading Whisper encoder on %s", secondary)
        self._whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
        whisper_full = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL,
            torch_dtype=torch.float16,
        )
        self._whisper_encoder = whisper_full.get_encoder().to(secondary)
        del whisper_full  # only keep the encoder

        # 5. Load image processor for reference image conditioning
        self._image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        # 6. Load DiT on primary GPU with CPU offload
        log.info("Loading HuMo DiT on %s", primary)
        from diffusers import WanTransformer3DModel

        self._dit = WanTransformer3DModel.from_pretrained(
            HUMO_REPO,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        gpu_index = primary.index
        if gpu_index is not None:
            self._dit = self._dit.to(primary)
        else:
            log.warning("No GPU — HuMo DiT on CPU (extremely slow)")

        log.info("HuMo engine loaded (model_size=%s)", self.config.model_size.value)

    def generate(self, input: VideoInput) -> VideoResult:
        """
        Generate a single video clip in TIA mode.

        Steps:
        1. Encode text prompt → text embeddings via T5
        2. Process reference image → image conditioning
        3. Process audio → Whisper audio embeddings
        4. Run denoising loop with dual CFG (text + audio guidance)
        5. VAE decode → frames tensor
        6. Save frames as MP4 (25fps, no audio)
        """
        if not self.is_loaded:
            raise RuntimeError("HumoEngine not loaded. Call load() first.")

        import torchaudio
        from PIL import Image

        primary = self.device_map.dit_device
        secondary = self.device_map.encoder_device

        # --- 1. Encode text ---
        log.info("Encoding text prompt...")
        text_inputs = self._t5_tokenizer(
            input.text_prompt,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(secondary)

        with torch.no_grad():
            text_embeds = self._t5_encoder(**text_inputs).last_hidden_state  # (1, seq, dim)

        # Unconditioned text embeddings for CFG
        uncond_inputs = self._t5_tokenizer(
            "",
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(secondary)
        with torch.no_grad():
            uncond_text_embeds = self._t5_encoder(**uncond_inputs).last_hidden_state

        text_embeds = text_embeds.to(primary)
        uncond_text_embeds = uncond_text_embeds.to(primary)

        # --- 2. Process reference image ---
        log.info("Processing reference image: %s", input.reference_image)
        ref_image = Image.open(input.reference_image).convert("RGB")
        image_inputs = self._image_processor(
            images=ref_image, return_tensors="pt"
        )
        image_cond = image_inputs["pixel_values"].to(dtype=torch.bfloat16, device=primary)

        # --- 3. Process audio ---
        log.info("Processing audio: %s", input.audio_segment)
        waveform, sr = torchaudio.load(str(input.audio_segment))
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        whisper_inputs = self._whisper_processor(
            waveform.squeeze(0).numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        whisper_features = whisper_inputs.input_features.to(
            dtype=torch.float16, device=secondary
        )
        with torch.no_grad():
            audio_embeds = self._whisper_encoder(whisper_features).last_hidden_state
        audio_embeds = audio_embeds.to(dtype=torch.bfloat16, device=primary)

        # Null audio for CFG
        null_audio = torch.zeros_like(audio_embeds)

        # --- 4. Calculate frame count ---
        audio_duration = waveform.shape[-1] / 16000
        num_frames = min(int(audio_duration * FPS), MAX_FRAMES)
        num_frames = max(num_frames, 1)

        # --- 5. Denoising loop ---
        log.info("Running denoising (%d steps, %d frames)...", self.config.denoising_steps, num_frames)

        # Initialize latent noise
        latent_channels = self._dit.config.in_channels
        h = self.config.height // 8  # VAE downscale factor
        w = self.config.width // 8
        t_latent = (num_frames - 1) // 4 + 1  # temporal downscale

        latents = torch.randn(
            1, latent_channels, t_latent, h, w,
            device=primary, dtype=torch.bfloat16,
        )

        self._scheduler.set_timesteps(self.config.denoising_steps, device=primary)
        timesteps = self._scheduler.timesteps

        for i, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0).to(primary)

            # Conditional prediction (text + audio + image)
            with torch.no_grad():
                noise_pred_cond = self._dit(
                    latents,
                    timestep=t_batch,
                    encoder_hidden_states=text_embeds,
                    humo_audio_embeds=audio_embeds,
                    humo_image_cond=image_cond,
                ).sample

            # Unconditional prediction (for CFG)
            with torch.no_grad():
                noise_pred_uncond = self._dit(
                    latents,
                    timestep=t_batch,
                    encoder_hidden_states=uncond_text_embeds,
                    humo_audio_embeds=null_audio,
                ).sample

            # Dual classifier-free guidance
            noise_pred = noise_pred_uncond
            noise_pred = noise_pred + self.config.scale_t * (noise_pred_cond - noise_pred_uncond)

            latents = self._scheduler.step(noise_pred, t, latents).prev_sample

        # --- 6. VAE decode ---
        log.info("Decoding latents via VAE...")
        latents = latents.to(dtype=torch.float32, device=self.device_map.vae_device)
        with torch.no_grad():
            frames = self._vae.decode(latents).sample  # (1, C, T, H, W)

        # --- 7. Save as MP4 ---
        frames = frames.squeeze(0)  # (C, T, H, W)
        frames = frames.permute(1, 2, 3, 0)  # (T, H, W, C)
        frames = frames.clamp(-1, 1) * 0.5 + 0.5  # [-1,1] → [0,1]
        frames = (frames * 255).to(torch.uint8).cpu()

        _save_frames_as_mp4(frames, input.output_path, fps=FPS)

        actual_duration = num_frames / FPS
        log.info(
            "Generated clip: %s (%d frames, %.2fs)",
            input.output_path, num_frames, actual_duration,
        )

        return VideoResult(
            video_path=input.output_path,
            frames_generated=num_frames,
            duration_seconds=actual_duration,
            metadata={
                "scale_t": self.config.scale_t,
                "scale_a": self.config.scale_a,
                "denoising_steps": self.config.denoising_steps,
            },
        )

    def generate_scene(
        self,
        text_prompt: str,
        reference_image: Path,
        audio_segment: Path,
        output_dir: Path,
        scene_id: str,
        duration: float,
    ) -> list[VideoResult]:
        """
        Generate video for a full scene, handling sub-clips if duration > 3.88s.

        For scenes > MAX_DURATION:
          1. Split audio into sub-segments
          2. Generate each sub-clip with same reference image
          3. Return list of results

        Returns: list of VideoResult (one per sub-clip, or single item for short scenes)
        """
        if not self.is_loaded:
            raise RuntimeError("HumoEngine not loaded. Call load() first.")

        output_dir.mkdir(parents=True, exist_ok=True)

        if duration <= MAX_DURATION:
            # Single clip — no splitting needed
            output_path = output_dir / f"{scene_id}.mp4"
            result = self.generate(VideoInput(
                text_prompt=text_prompt,
                reference_image=reference_image,
                audio_segment=audio_segment,
                output_path=output_path,
            ))
            return [result]

        # Split into sub-clips
        num_clips = math.ceil(duration / MAX_DURATION)
        results: list[VideoResult] = []
        suffixes = _sub_clip_suffixes(num_clips)

        for i in range(num_clips):
            suffix = suffixes[i]
            sub_id = f"{scene_id}_{suffix}"
            clip_start = i * MAX_DURATION
            clip_end = min((i + 1) * MAX_DURATION, duration)

            # Slice audio for this sub-clip
            sub_audio = output_dir / f"{sub_id}_audio.wav"
            slice_audio(audio_segment, sub_audio, clip_start, clip_end)

            output_path = output_dir / f"{sub_id}.mp4"
            log.info(
                "Generating sub-clip %s (%.2fs–%.2fs)",
                sub_id, clip_start, clip_end,
            )

            result = self.generate(VideoInput(
                text_prompt=text_prompt,
                reference_image=reference_image,
                audio_segment=sub_audio,
                output_path=output_path,
            ))
            results.append(result)

            # Clean up temporary audio slice
            sub_audio.unlink(missing_ok=True)

        return results

    def unload(self) -> None:
        """Unload all model components and free VRAM."""
        for attr in (
            "_dit", "_t5_encoder", "_t5_tokenizer", "_vae",
            "_whisper_encoder", "_whisper_processor", "_image_processor",
            "_scheduler",
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        clear_vram()
        log.info("HuMo engine unloaded")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sub_clip_suffixes(n: int) -> list[str]:
    """Generate sub-clip suffixes: a, b, c, ... aa, ab, ..."""
    suffixes = []
    for i in range(n):
        if i < 26:
            suffixes.append(chr(ord("a") + i))
        else:
            suffixes.append(chr(ord("a") + i // 26 - 1) + chr(ord("a") + i % 26))
    return suffixes


def _save_frames_as_mp4(frames: torch.Tensor, output_path: Path, fps: int = 25) -> None:
    """
    Save a (T, H, W, C) uint8 tensor as MP4 using ffmpeg.

    Uses raw video pipe to ffmpeg to avoid torchvision.io dependency issues.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, c = frames.shape

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",  # no audio
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.stdin.write(frames.numpy().tobytes())
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        raise RuntimeError(f"ffmpeg failed saving video: {stderr}")
