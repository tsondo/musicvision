"""
CLI entry point for MusicVision.

Usage:
    musicvision create <directory> --name "My Video"
    musicvision serve <directory> [--port 8000]
    musicvision info <directory>
    musicvision detect-hardware
    musicvision download-weights --tier fp8_scaled
    musicvision generate-video --project ./my_video [--tier gguf_q4] [--block-swap 20]

Environment variables (LLM backend, API keys, etc.) are loaded from a .env
file in the current working directory if one exists.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
import uvicorn


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_create(args: argparse.Namespace) -> None:
    from musicvision.project import ProjectService

    project_dir = Path(args.directory).resolve()
    svc = ProjectService.create(project_dir, name=args.name)
    print(f"✓ Created project '{svc.config.name}' at {project_dir}")


def cmd_serve(args: argparse.Namespace) -> None:
    from musicvision.api.app import app, mount_project_files
    from musicvision.project import ProjectService

    project_dir = Path(args.directory).resolve()

    import musicvision.api.app as api_module

    api_module._project = ProjectService.open(project_dir)
    mount_project_files(project_dir)

    print(f"✓ Serving project '{api_module._project.config.name}'")
    print(f"  API:      http://localhost:{args.port}/docs")
    print(f"  Project:  {project_dir}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


def cmd_info(args: argparse.Namespace) -> None:
    from musicvision.project import ProjectService

    project_dir = Path(args.directory).resolve()
    svc = ProjectService.open(project_dir)

    print(f"Project:    {svc.config.name}")
    print(f"Directory:  {project_dir}")
    print(f"Audio:      {svc.config.song.audio_file or '(none)'}")
    print(f"BPM:        {svc.config.song.bpm or '(not detected)'}")
    print(f"Duration:   {svc.config.song.duration_seconds or '(unknown)'}s")
    print(f"Scenes:     {len(svc.scenes.scenes)}")
    print(f"HuMo:       tier={svc.config.humo.tier.value} ({svc.config.humo.model_size}) @ {svc.config.humo.resolution}")
    print(f"FLUX:       {svc.config.flux.model.value}, {svc.config.flux.effective_steps} steps")

    if svc.scenes.scenes:
        approved_img = sum(1 for s in svc.scenes.scenes if s.image_status == "approved")
        approved_vid = sum(1 for s in svc.scenes.scenes if s.video_status == "approved")
        print(f"  Images:   {approved_img}/{len(svc.scenes.scenes)} approved")
        print(f"  Videos:   {approved_vid}/{len(svc.scenes.scenes)} approved")


def cmd_detect_hardware(args: argparse.Namespace) -> None:
    """Print GPU info and recommended HuMo tier."""
    try:
        from musicvision.utils.gpu import detect_devices, recommend_tier, vram_info
        from musicvision.models import TIER_VRAM_GB
    except ImportError as exc:
        print(f"Error importing GPU module: {exc}")
        print("Ensure torch is installed: pip install torch")
        sys.exit(1)

    try:
        device_map = detect_devices()
        gpus = vram_info()
    except Exception as exc:
        print(f"CUDA detection failed: {exc}")
        print("No GPUs available — CPU-only mode")
        return

    if not gpus:
        print("No CUDA GPUs detected.")
        return

    print("GPU Configuration")
    print("-" * 50)
    for gpu in gpus:
        print(
            f"  GPU {gpu['index']}: {gpu['name']}\n"
            f"    Total: {gpu['total_gb']:.1f} GB  "
            f"Free: {gpu['free_gb']:.1f} GB  "
            f"Compute: {gpu['compute_capability']}"
        )

    tier = recommend_tier(device_map)
    vram_needed = TIER_VRAM_GB.get(tier.value, 0)
    print(f"\nRecommended HuMo tier: {tier.value}")
    print(f"  DiT VRAM requirement: ~{vram_needed:.0f} GB")
    print(f"  To use: set humo.tier = {tier.value} in project.yaml")
    print(f"  Or:     musicvision generate-video --project DIR --tier {tier.value}")


def cmd_download_weights(args: argparse.Namespace) -> None:
    """Download HuMo weights for a tier."""
    import os
    from musicvision.models import HumoTier
    from musicvision.video.weight_registry import download_all_for_tier, weight_status

    try:
        tier = HumoTier(args.tier)
    except ValueError:
        valid = [t.value for t in HumoTier]
        print(f"Unknown tier '{args.tier}'. Valid tiers: {', '.join(valid)}")
        sys.exit(1)

    hf_token = (
        args.token
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    if not hf_token:
        print(
            "No HuggingFace token found.\n"
            "Set HUGGINGFACE_TOKEN in .env or pass --token <token>.\n"
            "HuMo weights are gated — you need a token with accepted terms."
        )
        sys.exit(1)

    base_dir = Path(args.dir) if args.dir else None

    print(f"Checking weight status for tier '{tier.value}'…")
    status = weight_status(tier, base_dir)
    missing = [k for k, present in status.items() if not present]
    if not missing:
        print(f"All weights for tier '{tier.value}' are already present.")
        return

    print(f"Missing: {missing}")
    print(f"Downloading weights for tier '{tier.value}'…")
    paths = download_all_for_tier(tier, base_dir=base_dir, hf_token=hf_token)
    for key, path in paths.items():
        print(f"  ✓ {key}: {path}")
    print("Download complete.")


def cmd_generate_video(args: argparse.Namespace) -> None:
    """Generate video clips for a project."""
    from musicvision.models import HumoTier
    from musicvision.project import ProjectService
    from musicvision.utils.gpu import detect_devices
    from musicvision.video.humo_engine import HumoEngine, HumoInput
    from musicvision.video.prompt_generator import generate_video_prompts_batch

    project_dir = Path(args.project).resolve()
    svc = ProjectService.open(project_dir)

    # Override tier/block-swap if specified on command line
    if args.tier:
        try:
            svc.config.humo.tier = HumoTier(args.tier)
        except ValueError:
            valid = [t.value for t in HumoTier]
            print(f"Unknown tier '{args.tier}'. Valid tiers: {', '.join(valid)}")
            sys.exit(1)
    if args.block_swap is not None:
        svc.config.humo.block_swap_count = args.block_swap

    scenes = svc.scenes.scenes
    targets = (
        [s for s in scenes if s.id in args.scene_ids]
        if args.scene_ids
        else [s for s in scenes if s.video_status.value != "approved"]
    )

    if not targets:
        print("No scenes to process.")
        return

    # Generate prompts for scenes that need them
    needs_prompts = [s for s in targets if not s.effective_video_prompt]
    if needs_prompts:
        print(f"Generating video prompts for {len(needs_prompts)} scene(s)…")
        generate_video_prompts_batch(needs_prompts, svc.config.style_sheet, config=svc.config)
        svc.save_scenes()

    device_map = detect_devices()
    engine = HumoEngine(svc.config.humo, device_map)

    print(
        f"Loading HuMo engine (tier={svc.config.humo.tier.value}, "
        f"block_swap={svc.config.humo.block_swap_count})…"
    )
    engine.load()

    generated = 0
    errors = []
    try:
        for scene in targets:
            if not scene.reference_image:
                print(f"  SKIP {scene.id}: no reference image")
                errors.append(scene.id)
                continue
            if not scene.audio_segment:
                print(f"  SKIP {scene.id}: no audio segment")
                errors.append(scene.id)
                continue

            prompt = scene.effective_video_prompt or scene.effective_image_prompt or scene.lyrics or f"Scene {scene.id}"
            ref_image = svc.resolve_path(scene.reference_image)
            audio_seg = svc.resolve_path(scene.audio_segment)

            print(f"  Generating {scene.id} ({scene.duration:.2f}s)…")
            try:
                outputs = engine.generate_scene(
                    text_prompt=prompt,
                    reference_image=ref_image,
                    audio_segment=audio_seg,
                    output_dir=svc.paths.clips_dir,
                    scene_id=scene.id,
                    duration=scene.duration,
                )
                if outputs:
                    if len(outputs) == 1:
                        scene.video_clip = str(outputs[0].video_path.relative_to(svc.paths.root))
                    else:
                        from musicvision.models import SubClip, ApprovalStatus
                        scene.sub_clips = []
                        for i, out in enumerate(outputs):
                            sc = SubClip(
                                id=f"{scene.id}_sub_{i:02d}",
                                time_start=scene.time_start + i * 3.88,
                                time_end=min(scene.time_start + (i + 1) * 3.88, scene.time_end),
                                video_clip=str(out.video_path.relative_to(svc.paths.root)),
                            )
                            scene.sub_clips.append(sc)
                    from musicvision.models import ApprovalStatus
                    scene.video_status = ApprovalStatus.PENDING
                    generated += 1
                    print(f"  ✓ {scene.id} → {scene.video_clip or f'{len(outputs)} sub-clips'}")
            except Exception as exc:
                print(f"  ✗ {scene.id}: {exc}")
                errors.append(scene.id)
    finally:
        engine.unload()

    svc.save_scenes()
    print(f"\nGenerated {generated}/{len(targets)} clips.")
    if errors:
        print(f"Failed: {errors}")


def main() -> None:
    load_dotenv()  # load .env from cwd before any env vars are read

    parser = argparse.ArgumentParser(prog="musicvision", description="AI music video production pipeline")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    # create
    p_create = sub.add_parser("create", help="Create a new project")
    p_create.add_argument("directory", help="Project directory path")
    p_create.add_argument("--name", default="Untitled Project", help="Project name")

    # serve
    p_serve = sub.add_parser("serve", help="Start the API server for a project")
    p_serve.add_argument("directory", help="Project directory path")
    p_serve.add_argument("--port", type=int, default=8000, help="Port to serve on")

    # info
    p_info = sub.add_parser("info", help="Show project info")
    p_info.add_argument("directory", help="Project directory path")

    # detect-hardware
    sub.add_parser("detect-hardware", help="Detect GPUs and print recommended HuMo tier")

    # download-weights
    p_dl = sub.add_parser("download-weights", help="Download HuMo weights for a tier")
    p_dl.add_argument(
        "--tier", required=True,
        choices=[t.value for t in __import__("musicvision.models", fromlist=["HumoTier"]).HumoTier],
        help="Precision tier to download",
    )
    p_dl.add_argument("--token", default=None, help="HuggingFace token (overrides env var)")
    p_dl.add_argument("--dir", default=None, help="Override weights directory")

    # generate-video
    p_gv = sub.add_parser("generate-video", help="Generate video clips for a project")
    p_gv.add_argument("--project", required=True, help="Project directory path")
    p_gv.add_argument(
        "--tier", default=None,
        help="Override HuMo tier (fp16/fp8_scaled/gguf_q8/gguf_q6/gguf_q4/preview)",
    )
    p_gv.add_argument("--block-swap", type=int, default=None, dest="block_swap",
                      help="Override block swap count (0 = all on GPU)")
    p_gv.add_argument("--scene-ids", nargs="*", default=[], dest="scene_ids",
                      help="Scene IDs to generate (default: all non-approved)")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "create":
        cmd_create(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "detect-hardware":
        cmd_detect_hardware(args)
    elif args.command == "download-weights":
        cmd_download_weights(args)
    elif args.command == "generate-video":
        cmd_generate_video(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
