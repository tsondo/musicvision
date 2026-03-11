"""Test HuMo inference at 480p with fixed zero_vae."""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from musicvision.video.humo_engine import HumoEngine, HumoInput
from musicvision.models import HumoConfig, HumoTier, VideoEngineType
from musicvision.utils.gpu import detect_devices
from musicvision.video.factory import create_video_engine

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# 480p matches the official zero_vae_129frame.pt (lat 60x104)
config = HumoConfig(
    tier=HumoTier.FP8_SCALED,
    resolution="480p",
    denoising_steps=30,  # enough steps for meaningful output
    shift=5.0,
    scale_a=5.5,
    scale_t=5.0,
    sampler="uni_pc",
)

print(f"Config: {config.resolution} ({config.width}x{config.height}), "
      f"{config.denoising_steps} steps, sampler={config.sampler}")

device_map = detect_devices()
engine = create_video_engine(config, device_map=device_map, engine_type=VideoEngineType.HUMO)

print("Loading engine...")
engine.load()
print("Engine loaded.")

ref_img = Path("test_output/2026-03-01_1430_full_llm/images/scene_001.png")
if not ref_img.exists():
    print(f"Reference image not found: {ref_img}")
    sys.exit(1)

# 1s silent audio
import subprocess
short_audio = Path("/tmp/humo_test_1s.wav")
subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1.0",
                str(short_audio)], capture_output=True)

inp = HumoInput(
    reference_image=ref_img,
    audio_segment=short_audio,
    seed=42,
    output_path=Path("/tmp/humo_480p_zerovae_fix.mp4"),
    text_prompt="A woman lying in bed at dawn, soft light through window",
)

print(f"Generating at {config.width}x{config.height}...")
try:
    result = engine.generate(inp)
    print(f"Result: {result.video_path}")
    print(f"Frames: {result.n_frames}")

    # Extract frames for inspection
    for frame_n in [0, 5, 10, 15, 20]:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(result.video_path),
            "-vf", f"select=eq(n\\,{frame_n})", "-vframes", "1",
            f"/tmp/humo_480p_fix_frame{frame_n:02d}.png"
        ], capture_output=True)
    print("Frames saved to /tmp/humo_480p_fix_frame*.png")
except Exception as e:
    print(f"Generation failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    engine.unload()
    print("Engine unloaded.")
