#!/usr/bin/env python3
"""
HunyuanVideo-Avatar subprocess wrapper.

Bridge script that runs inside the HVA virtual environment.  The MusicVision
engine class calls this via subprocess with:

    python scripts/hva_wrapper.py --request /path/to/request.json

Request JSON schema::

    {
        "image_path":       "/abs/path/to/reference.png",
        "audio_path":       "/abs/path/to/audio.wav",
        "prompt":           "A woman singing in a studio",
        "output_path":      "/abs/path/to/output.mp4",
        "hva_repo_dir":     "/home/user/HunyuanVideoAvatar",
        "checkpoint":       "fp8",
        "image_size":       704,
        "sample_n_frames":  129,
        "cfg_scale":        7.5,
        "infer_steps":      50,
        "flow_shift":       5.0,
        "seed":             42,
        "use_deepcache":    true,
        "use_fp8":          true,
        "cpu_offload":      true
    }

Response JSON (written to --response path)::

    {
        "status":           "success",
        "video_path":       "/abs/path/to/output.mp4",
        "frames_generated": 129,
        "duration":         5.16,
        "error":            null
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def build_csv(
    image_path: str, audio_path: str, prompt: str, csv_path: Path, fps: int = 25,
) -> None:
    """Write a single-row CSV in HVA's expected format.

    Required columns: videoid, image, audio, prompt, fps
    """
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["videoid", "image", "audio", "prompt", "fps"])
        writer.writerow(["clip_0", image_path, audio_path, prompt, fps])


def run_hva(request: dict) -> dict:
    """Execute sample_gpu_poor.py and return a response dict."""
    hva_repo = Path(request["hva_repo_dir"])
    sample_script = hva_repo / "hymm_sp" / "sample_gpu_poor.py"

    if not sample_script.exists():
        return {
            "status": "error",
            "video_path": None,
            "frames_generated": 0,
            "duration": 0.0,
            "error": f"sample_gpu_poor.py not found at {sample_script}",
        }

    weights_dir = hva_repo / "weights"
    ckpt_dir = weights_dir / "ckpts" / "hunyuan-video-t2v-720p" / "transformers"
    checkpoint = request.get("checkpoint", "bf16")
    if checkpoint == "fp8":
        ckpt_path = ckpt_dir / "mp_rank_00_model_states_fp8.pt"
    else:
        ckpt_path = ckpt_dir / "mp_rank_00_model_states.pt"

    if not ckpt_path.exists():
        return {
            "status": "error",
            "video_path": None,
            "frames_generated": 0,
            "duration": 0.0,
            "error": f"Checkpoint not found at {ckpt_path}",
        }

    with tempfile.TemporaryDirectory(prefix="hva_") as tmpdir:
        tmpdir = Path(tmpdir)
        csv_path = tmpdir / "input.csv"
        save_path = tmpdir / "output"
        save_path.mkdir()

        build_csv(
            request["image_path"],
            request["audio_path"],
            request.get("prompt", ""),
            csv_path,
        )

        image_size = request.get("image_size", 704)
        sample_n_frames = request.get("sample_n_frames", 129)
        cfg_scale = request.get("cfg_scale", 7.5)
        infer_steps = request.get("infer_steps", 50)
        flow_shift = request.get("flow_shift", 5.0)
        seed = request.get("seed")

        cmd = [
            sys.executable,
            str(sample_script),
            "--input", str(csv_path),
            "--ckpt", str(ckpt_path),
            "--sample-n-frames", str(sample_n_frames),
            "--image-size", str(image_size),
            "--cfg-scale", str(cfg_scale),
            "--infer-steps", str(infer_steps),
            "--flow-shift-eval-video", str(flow_shift),
            "--save-path", str(save_path),
        ]

        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        if request.get("use_deepcache", True):
            cmd.extend(["--use-deepcache", "1"])

        if request.get("use_fp8", False):
            cmd.append("--use-fp8")

        if request.get("cpu_offload", True):
            cmd.append("--cpu-offload")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(hva_repo)
        env["DISABLE_SP"] = "1"
        env["MODEL_BASE"] = str(weights_dir)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(hva_repo),
                env=env,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout per clip (704p@30 steps ≈ 15 min)
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "video_path": None,
                "frames_generated": 0,
                "duration": 0.0,
                "error": "HVA inference timed out (600s)",
            }

        if result.returncode != 0:
            return {
                "status": "error",
                "video_path": None,
                "frames_generated": 0,
                "duration": 0.0,
                "error": f"HVA exited with code {result.returncode}: {result.stderr[-2000:]}",
            }

        # Find the generated video in save_path
        videos = list(save_path.rglob("*.mp4"))
        if not videos:
            return {
                "status": "error",
                "video_path": None,
                "frames_generated": 0,
                "duration": 0.0,
                "error": f"No MP4 found in output dir. stdout: {result.stdout[-1000:]}",
            }

        src_video = videos[0]
        output_path = Path(request["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_video), str(output_path))

        fps = 25
        duration = sample_n_frames / fps

        return {
            "status": "success",
            "video_path": str(output_path),
            "frames_generated": sample_n_frames,
            "duration": duration,
            "error": None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="HunyuanVideo-Avatar subprocess wrapper")
    parser.add_argument("--request", required=True, help="Path to request JSON file")
    parser.add_argument("--response", required=True, help="Path to write response JSON file")
    args = parser.parse_args()

    with open(args.request) as f:
        request = json.load(f)

    response = run_hva(request)

    with open(args.response, "w") as f:
        json.dump(response, f, indent=2)

    # Exit with non-zero if error
    if response["status"] != "success":
        print(f"HVA error: {response['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
