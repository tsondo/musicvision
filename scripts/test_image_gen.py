#!/usr/bin/env python3
"""
GPU integration test for image generation engines (FLUX + Z-Image).

Generates 2 images per engine (man + woman) at HVA-compatible resolution.
Saves results to test_output/<timestamp>_imagegen/.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# Ensure musicvision is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from musicvision.imaging import create_engine
from musicvision.models import FluxQuant, ImageGenConfig, ImageModel
from musicvision.utils.gpu import DeviceMap, detect_devices

# ---------- Config ----------
# Reference image resolution — will be resized by video engine anyway.
# 768x512 fits in 32GB VRAM with Z-Image 6B + CPU offload.
WIDTH = 768
HEIGHT = 512

PROMPTS = {
    "woman": "A woman with long dark hair standing in a recording studio, soft lighting, photorealistic",
    "man": "A man with short hair wearing a leather jacket on a city rooftop at dusk, photorealistic",
}

# ---------- Main ----------
def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent.parent / "test_output" / f"{stamp}_imagegen"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    device_map = detect_devices()
    print(f"Detected: primary={device_map.primary}, secondary={device_map.secondary}")

    # For image gen, use the biggest GPU for everything (avoids cross-device tensor issues)
    import torch
    biggest = device_map.primary
    if device_map.secondary and device_map.secondary.type != "cpu":
        mem_p = torch.cuda.get_device_properties(device_map.primary).total_memory
        mem_s = torch.cuda.get_device_properties(device_map.secondary).total_memory
        if mem_s > mem_p:
            biggest = device_map.secondary
    device_map = DeviceMap(
        dit_device=biggest, encoder_device=biggest,
        vae_device=biggest, offload_device=torch.device("cpu"),
    )
    print(f"Using: {biggest} for all components")

    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    engines: list[tuple[str, ImageGenConfig]] = []

    # Z-Image-Turbo (ungated, no token needed)
    engines.append(("z-image-turbo", ImageGenConfig(model=ImageModel.ZIMAGE_TURBO)))

    # FLUX-schnell (gated, needs accepted license + token)
    # Force BF16 quant to use cpu_offload strategy (T5-XXL doesn't fit on secondary GPU)
    if hf_token:
        engines.append(("flux-schnell", ImageGenConfig(
            model=ImageModel.FLUX_SCHNELL, quant=FluxQuant.BF16,
        )))
    else:
        print("HUGGINGFACE_TOKEN not set — skipping FLUX (gated model)")

    for engine_name, config in engines:
        engine_dir = out_dir / engine_name
        engine_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Engine: {engine_name} (steps={config.effective_steps}, guidance={config.guidance_scale})")
        print(f"{'='*60}")

        engine = create_engine(config, device_map)

        print("Loading model...")
        t0 = time.time()
        try:
            engine.load()
        except Exception as e:
            print(f"FAILED to load {engine_name}: {e}")
            continue
        print(f"Loaded in {time.time() - t0:.1f}s")

        for label, prompt in PROMPTS.items():
            out_path = engine_dir / f"{label}.png"
            print(f"\nGenerating {label}...")
            print(f"  Prompt: {prompt[:80]}...")
            t0 = time.time()
            try:
                result = engine.generate(
                    prompt=prompt,
                    output_path=out_path,
                    width=WIDTH,
                    height=HEIGHT,
                    seed=42,
                )
                elapsed = time.time() - t0
                print(f"  Saved: {result.path} ({result.width}x{result.height})")
                print(f"  Seed: {result.seed}, Time: {elapsed:.1f}s")
            except Exception as e:
                print(f"  FAILED: {e}")

        print(f"\nUnloading {engine_name}...")
        engine.unload()
        print("Done.")

    print(f"\n{'='*60}")
    print(f"All images saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
