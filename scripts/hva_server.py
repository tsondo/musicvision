#!/usr/bin/env python3
"""
Persistent HunyuanVideo-Avatar server process.

Loads HVA models once and processes clips via stdin/stdout JSON-line IPC.
This avoids the ~7 min model reload per clip when using hva_wrapper.py.

Protocol
--------
1. Startup: Load all models (sampler, whisper, face alignment, feature extractor)
2. Ready signal: Write ``{"status": "ready"}`` to stdout
3. Request loop: Read JSON line from stdin → process clip → write JSON line to stdout
4. Shutdown: EOF or ``{"command": "shutdown"}`` → clean exit

CRITICAL: stdout is captured for JSON IPC before any HVA imports. All HVA
debug prints are redirected to stderr via ``sys.stdout = sys.stderr``.

Invoked by HunyuanAvatarEngine.load() as::

    <hva_venv_python> scripts/hva_server.py --ckpt <path> --image-size 704 ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Capture real stdout BEFORE anything else can print to it
# ---------------------------------------------------------------------------
_output_fd = os.dup(sys.stdout.fileno())
_output = os.fdopen(_output_fd, "w")
sys.stdout = sys.stderr  # All HVA prints go to stderr


def send(msg: dict) -> None:
    """Write a JSON line to the real stdout (IPC channel)."""
    _output.write(json.dumps(msg) + "\n")
    _output.flush()


def send_error(error: str) -> None:
    send({"status": "error", "video_path": None, "frames_generated": 0, "duration": 0.0, "error": error})


# ---------------------------------------------------------------------------
# 2. Parse args
# ---------------------------------------------------------------------------

def build_args():
    parser = argparse.ArgumentParser(description="Persistent HVA server")
    parser.add_argument("--hva-repo-dir", required=True, help="Path to HunyuanVideoAvatar repo")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint file")
    parser.add_argument("--image-size", type=int, default=704)
    parser.add_argument("--sample-n-frames", type=int, default=129)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--infer-steps", type=int, default=50)
    parser.add_argument("--flow-shift-eval-video", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-deepcache", type=int, default=1)
    parser.add_argument("--use-fp8", action="store_true", default=False)
    parser.add_argument("--cpu-offload", action="store_true", default=False)
    return parser.parse_args()


def main():
    server_args = build_args()
    hva_repo = Path(server_args.hva_repo_dir)
    weights_dir = hva_repo / "weights"

    # ---------------------------------------------------------------------------
    # 3. Import HVA modules (add repo to sys.path)
    # ---------------------------------------------------------------------------
    sys.path.insert(0, str(hva_repo))
    os.environ["PYTHONPATH"] = str(hva_repo)
    os.environ["DISABLE_SP"] = "1"
    os.environ["MODEL_BASE"] = str(weights_dir)

    try:
        import torch
        import numpy as np
        import imageio
        from einops import rearrange

        from hymm_sp.config import parse_args
        from hymm_sp.sample_inference_audio import HunyuanVideoSampler
        from hymm_sp.data_kits.audio_dataset import VideoAudioTextLoaderVal
        from hymm_sp.data_kits.face_align import AlignImage
        from transformers import WhisperModel, AutoFeatureExtractor
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
    except Exception as exc:
        send({"status": "error", "error": f"Failed to import HVA modules: {exc}"})
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # 4. Load models — this is the expensive part (~7 min)
    # ---------------------------------------------------------------------------
    print(f"[hva_server] Loading models from {server_args.ckpt}", file=sys.stderr)

    try:
        # Build HVA args by passing the same CLI flags through HVA's parse_args
        hva_argv = [
            "--input", "/dev/null",  # placeholder, overridden per clip
            "--ckpt", server_args.ckpt,
            "--sample-n-frames", str(server_args.sample_n_frames),
            "--image-size", str(server_args.image_size),
            "--cfg-scale", str(server_args.cfg_scale),
            "--infer-steps", str(server_args.infer_steps),
            "--flow-shift-eval-video", str(server_args.flow_shift_eval_video),
            "--save-path", "/tmp/hva_server_dummy",
        ]
        if server_args.seed is not None:
            hva_argv.extend(["--seed", str(server_args.seed)])
        if server_args.use_deepcache:
            hva_argv.extend(["--use-deepcache", "1"])
        if server_args.use_fp8:
            hva_argv.append("--use-fp8")
        if server_args.cpu_offload:
            hva_argv.append("--cpu-offload")

        # Monkey-patch sys.argv so HVA's parse_args picks up our flags
        old_argv = sys.argv
        sys.argv = ["hva_server"] + hva_argv
        hva_args = parse_args()
        sys.argv = old_argv

        device = torch.device("cuda")
        sampler = HunyuanVideoSampler.from_pretrained(
            server_args.ckpt, args=hva_args, device=device,
        )
        hva_args = sampler.args

        if hva_args.cpu_offload:
            from diffusers.hooks import apply_group_offloading
            apply_group_offloading(
                sampler.pipeline.transformer,
                onload_device=device,
                offload_type="block_level",
                num_blocks_per_group=1,
            )

        wav2vec = WhisperModel.from_pretrained(
            str(weights_dir / "ckpts" / "whisper-tiny"),
        ).to(device=device, dtype=torch.float32)
        wav2vec.requires_grad_(False)

        det_path = str(weights_dir / "ckpts" / "det_align" / "detface.pt")
        align_instance = AlignImage("cuda", det_path=det_path)

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            str(weights_dir / "ckpts" / "whisper-tiny"),
        )

        print("[hva_server] All models loaded successfully", file=sys.stderr)
    except Exception as exc:
        send({"status": "error", "error": f"Failed to load models: {exc}"})
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # 5. Ready signal
    # ---------------------------------------------------------------------------
    send({"status": "ready"})

    # ---------------------------------------------------------------------------
    # 6. Request loop
    # ---------------------------------------------------------------------------
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            send_error(f"Invalid JSON: {exc}")
            continue

        if request.get("command") == "shutdown":
            print("[hva_server] Shutdown requested", file=sys.stderr)
            break

        try:
            result = process_clip(
                request, hva_args, sampler, wav2vec, feature_extractor,
                align_instance, device,
            )
            send(result)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            send_error(f"Inference failed: {exc}")

    print("[hva_server] Exiting", file=sys.stderr)


def process_clip(
    request: dict,
    hva_args,
    sampler,
    wav2vec,
    feature_extractor,
    align_instance,
    device,
) -> dict:
    """Run inference for a single clip and return a response dict."""
    import torch
    import numpy as np
    import imageio
    from einops import rearrange
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from hymm_sp.data_kits.audio_dataset import VideoAudioTextLoaderVal

    image_path = request["image_path"]
    audio_path = request["audio_path"]
    prompt = request.get("prompt", "")
    output_path = Path(request["output_path"])
    sample_n_frames = request.get("sample_n_frames", 129)
    fps = request.get("fps", 25)

    with tempfile.TemporaryDirectory(prefix="hva_clip_") as tmpdir:
        tmpdir = Path(tmpdir)
        csv_path = tmpdir / "input.csv"
        save_path = tmpdir / "output"
        save_path.mkdir()

        # Write single-row CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["videoid", "image", "audio", "prompt", "fps"])
            writer.writerow(["clip_0", image_path, audio_path, prompt, fps])

        # Build dataset and dataloader (lightweight — no model loading)
        kwargs = {
            "text_encoder": sampler.text_encoder,
            "text_encoder_2": sampler.text_encoder_2,
            "feature_extractor": feature_extractor,
        }
        dataset = VideoAudioTextLoaderVal(
            image_size=hva_args.image_size,
            meta_file=str(csv_path),
            **kwargs,
        )
        ds_sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False, drop_last=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=ds_sampler, drop_last=False)

        for batch in loader:
            audio_clip_path = str(batch["audio_path"][0])

            if getattr(hva_args, "infer_min", False):
                batch["audio_len"][0] = 129

            samples = sampler.predict(hva_args, batch, wav2vec, feature_extractor, align_instance)

            sample = samples["samples"][0].unsqueeze(0)
            sample = sample[:, :, :batch["audio_len"][0]]

            video = rearrange(sample[0], "c f h w -> f h w c")
            video = (video * 255.0).data.cpu().numpy().astype(np.uint8)

            torch.cuda.empty_cache()

            final_frames = np.stack(list(video), axis=0)

            # Write video to temp, then mux with audio
            raw_path = save_path / "clip_0.mp4"
            output_audio_path = save_path / "clip_0_audio.mp4"
            imageio.mimsave(str(raw_path), final_frames, fps=fps)
            os.system(
                f"ffmpeg -i '{raw_path}' -i '{audio_clip_path}' -shortest '{output_audio_path}' -y -loglevel quiet; "
                f"rm '{raw_path}'"
            )

            # Move to final output path
            src = output_audio_path if output_audio_path.exists() else raw_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(output_path))

            duration = sample_n_frames / fps
            return {
                "status": "success",
                "video_path": str(output_path),
                "frames_generated": sample_n_frames,
                "duration": duration,
            }

    # Should not reach here, but just in case
    return {"status": "error", "video_path": None, "frames_generated": 0, "duration": 0.0, "error": "No batch in loader"}


if __name__ == "__main__":
    main()
