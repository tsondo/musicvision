#!/usr/bin/env python3
"""
Integration test for HunyuanVideo-Avatar via the MusicVision engine.

Requires:
  - HVA repo cloned at ~/HunyuanVideoAvatar with weights downloaded
  - HVA venv set up at ~/HunyuanVideoAvatar/.venv
  - GPU available (runs on CUDA_VISIBLE_DEVICES=0 by default)

Usage:
    python scripts/test_hva_standalone.py [--image PATH] [--audio PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musicvision.models import HunyuanAvatarConfig
from musicvision.video.hunyuan_avatar_engine import HunyuanAvatarEngine
from musicvision.video.base import VideoInput


def main() -> None:
    parser = argparse.ArgumentParser(description="Test HunyuanVideo-Avatar standalone")
    parser.add_argument("--image", default=None, help="Reference image path")
    parser.add_argument("--audio", default=None, help="Audio file path")
    parser.add_argument("--output", default="./hva_test_output.mp4", help="Output video path")
    parser.add_argument("--hva-dir", default=str(Path.home() / "HunyuanVideoAvatar"), help="HVA repo directory")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--frames", type=int, default=129, help="Number of frames")
    args = parser.parse_args()

    hva_dir = Path(args.hva_dir)
    venv_python = hva_dir / ".venv" / "bin" / "python"

    if not hva_dir.is_dir():
        print(f"HVA repo not found at {hva_dir}")
        sys.exit(1)
    if not venv_python.is_file():
        print(f"HVA venv python not found at {venv_python}")
        sys.exit(1)

    # Find test image
    if args.image:
        image = Path(args.image)
    else:
        # Try common test image locations
        candidates = [
            Path("tests/fixtures/test_woman.jpeg"),
            Path("test_woman.jpeg"),
        ]
        image = next((p for p in candidates if p.exists()), None)
        if image is None:
            print("No test image found. Pass --image PATH")
            sys.exit(1)

    # Find test audio
    if args.audio:
        audio = Path(args.audio)
    else:
        print("No audio file specified. Pass --audio PATH")
        sys.exit(1)

    print(f"Image:  {image}")
    print(f"Audio:  {audio}")
    print(f"Output: {args.output}")
    print(f"HVA:    {hva_dir}")
    print(f"Steps:  {args.steps}")
    print(f"Frames: {args.frames}")
    print()

    config = HunyuanAvatarConfig(
        hva_repo_dir=str(hva_dir),
        hva_venv_python=str(venv_python),
        infer_steps=args.steps,
        sample_n_frames=args.frames,
    )

    engine = HunyuanAvatarEngine(config)

    print("Validating HVA setup...")
    engine.load()
    print("Setup validated.")

    inp = VideoInput(
        text_prompt="A person speaking and gesturing naturally",
        reference_image=image.resolve(),
        audio_segment=audio.resolve(),
        output_path=Path(args.output).resolve(),
    )

    print("Starting generation...")
    t0 = time.time()
    result = engine.generate(inp)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Video: {result.video_path}")
    print(f"  Frames: {result.frames_generated}")
    print(f"  Duration: {result.duration_seconds:.2f}s")

    engine.unload()


if __name__ == "__main__":
    main()
