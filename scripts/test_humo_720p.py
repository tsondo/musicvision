"""Quick standalone test: HuMo inference at 720p (27 frames) to check for checkerboard."""
import sys
import torch
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from musicvision.video.humo_engine import HumoEngine
from musicvision.models import HumoConfig, HumoTier

# Config: 720p, 10 steps (just enough to see if output has structure)
config = HumoConfig(
    tier=HumoTier.FP8_SCALED,
    resolution="480p",
    denoising_steps=10,
    shift=5.0,
    scale_a=5.5,
    scale_t=5.0,
    sampler="euler",
)

print(f"Config: {config.resolution} ({config.width}x{config.height}), {config.denoising_steps} steps")

from musicvision.utils.gpu import detect_devices
from musicvision.video.factory import create_video_engine
from musicvision.models import VideoEngineType

device_map = detect_devices()
engine = create_video_engine(config, device_map=device_map, engine_type=VideoEngineType.HUMO)

print("Loading engine...")
engine.load()
print("Engine loaded.")

# Use the same reference image
ref_img = Path("test_output/2026-03-01_1430_full_llm/images/scene_001.png")
audio_seg = Path("test_output/2026-03-01_1430_full_llm/segments/scene_001.wav")

if not ref_img.exists():
    print(f"Reference image not found: {ref_img}")
    sys.exit(1)

# Create a short 1-second silent WAV for testing (to keep frames low at 720p)
import subprocess
short_audio = Path("/tmp/humo_test_1s.wav")
subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1.0",
                str(short_audio)], capture_output=True)
audio_seg = short_audio
print(f"Using 1s silent audio: {audio_seg}")

from musicvision.video.humo_engine import HumoInput
import random

seed = 42

inp = HumoInput(
    reference_image=ref_img,
    audio_segment=audio_seg,
    seed=seed,
    output_path=Path("/tmp/humo_720p_test.mp4"),
    text_prompt="A woman lying in bed at dawn, soft light through window",
)

print(f"Generating at {config.width}x{config.height} (frames determined by audio)...")

# Generate
try:
    result = engine.generate(inp)
    print(f"Result: {result.video_path}")
    print(f"Frames: {result.n_frames}")

    # Extract a frame for inspection
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", str(result.video_path),
        "-vf", "select=eq(n\\,10)", "-vframes", "1",
        "/tmp/humo_720p_frame10.png"
    ], capture_output=True)
    print("Frame saved to /tmp/humo_720p_frame10.png")
except Exception as e:
    print(f"Generation failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    engine.unload()
    print("Engine unloaded.")
