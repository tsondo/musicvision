#!/usr/bin/env python3
"""
SeedVR2 subprocess wrapper.

Bridge script that runs inside the SeedVR2 virtual environment.  The MusicVision
engine class calls this via subprocess with JSON on stdin/stdout:

    echo '{"video_path": "...", ...}' | python scripts/seedvr2_wrapper.py

Request JSON schema::

    {
        "video_path":      "/abs/path/to/input.mp4",
        "output_path":     "/abs/path/to/output.mp4",
        "target_width":    1920,
        "target_height":   1080,
        "model_id":        "ByteDance-Seed/SeedVR2-3B",
        "use_fp8":         true
    }

Response JSON (last line of stdout)::

    {
        "status":      "success",
        "video_path":  "/abs/path/to/output.mp4",
        "error":       null
    }
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_seedvr2(request: dict) -> dict:
    """Execute SeedVR2 inference and return a response dict."""
    video_path = Path(request["video_path"])
    output_path = Path(request["output_path"])
    target_w = request["target_width"]
    target_h = request["target_height"]

    if not video_path.exists():
        return {"status": "error", "video_path": None, "error": f"Input not found: {video_path}"}

    seedvr_repo = Path(__file__).resolve().parent.parent
    # If running from MusicVision scripts dir, SEEDVR2_REPO_DIR should be set
    seedvr_repo_env = os.environ.get("SEEDVR2_REPO_DIR")
    if seedvr_repo_env:
        seedvr_repo = Path(seedvr_repo_env)

    inference_script = seedvr_repo / "projects" / "inference_seedvr2_3b.py"
    if not inference_script.exists():
        return {
            "status": "error",
            "video_path": None,
            "error": f"Inference script not found: {inference_script}",
        }

    with tempfile.TemporaryDirectory(prefix="seedvr2_") as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input"
        output_dir = tmpdir / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # SeedVR2 reads all files from a folder — symlink our single video
        link = input_dir / video_path.name
        os.symlink(str(video_path), str(link))

        # Build torchrun command
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc-per-node=1",
            str(inference_script),
            "--video_path", str(input_dir),
            "--output_dir", str(output_dir),
            "--res_h", str(target_h),
            "--res_w", str(target_w),
            "--sp_size", "1",
            "--seed", "42",
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(seedvr_repo)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        print(f"Running SeedVR2: {video_path.name} → {target_w}x{target_h}", file=sys.stderr)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(seedvr_repo),
                env=env,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "video_path": None, "error": "SeedVR2 inference timed out (30m)"}

        if result.returncode != 0:
            stderr_tail = result.stderr[-2000:] if result.stderr else ""
            return {
                "status": "error",
                "video_path": None,
                "error": f"SeedVR2 exited with code {result.returncode}: {stderr_tail}",
            }

        # Find generated output
        outputs = list(output_dir.rglob("*.mp4")) + list(output_dir.rglob("*.png"))
        if not outputs:
            return {
                "status": "error",
                "video_path": None,
                "error": f"No output found. stdout: {result.stdout[-500:]}",
            }

        src = outputs[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # SeedVR2 outputs at its own latent-space resolution, not the exact target.
        # Final ffmpeg scale to hit the requested target resolution.
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            scaled = tmpdir / "scaled.mp4"
            scale_cmd = [
                ffmpeg, "-y", "-i", str(src),
                "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-an", str(scaled),
            ]
            scale_result = subprocess.run(scale_cmd, capture_output=True, text=True, timeout=120)
            if scale_result.returncode == 0 and scaled.exists():
                src = scaled
            else:
                print(f"ffmpeg scale failed, using SeedVR2 output as-is", file=sys.stderr)

        shutil.move(str(src), str(output_path))

        return {"status": "success", "video_path": str(output_path), "error": None}


def main() -> None:
    """Read JSON request from stdin, write JSON response to stdout."""
    raw = sys.stdin.read().strip()
    if not raw:
        response = {"status": "error", "video_path": None, "error": "Empty stdin"}
        print(json.dumps(response))
        sys.exit(1)

    try:
        request = json.loads(raw)
    except json.JSONDecodeError as e:
        response = {"status": "error", "video_path": None, "error": f"Invalid JSON: {e}"}
        print(json.dumps(response))
        sys.exit(1)

    response = run_seedvr2(request)
    print(json.dumps(response))

    if response["status"] != "success":
        print(f"SeedVR2 error: {response['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
