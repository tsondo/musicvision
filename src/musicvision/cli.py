"""
CLI entry point for MusicVision.

Usage:
    musicvision create <directory> --name "My Video"
    musicvision import-audio --project DIR --audio song.wav [--lyrics lyrics.txt]
    musicvision intake --project DIR [--llm] [--skip-transcription]
    musicvision generate-images --project DIR [--model z-image-turbo]
    musicvision generate-video --project DIR [--engine humo]
    musicvision assemble --project DIR [--approved-only]
    musicvision info <directory>
    musicvision serve <directory> [--port 8000]
    musicvision detect-hardware
    musicvision download-weights --tier fp8_scaled

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

    if args.directory:
        project_dir = Path(args.directory).resolve()

        import musicvision.api.app as api_module

        api_module._project = ProjectService.open(project_dir)
        mount_project_files(project_dir)

        print(f"✓ Serving project '{api_module._project.config.name}'")
        print(f"  Project:  {project_dir}")
    else:
        print("✓ Starting server with no project loaded")
        print("  Use the frontend to create or open a project")

    print(f"  API:      http://localhost:{args.port}/docs")
    print(f"  Frontend: http://localhost:5173")

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
    print(f"Video:      engine={svc.config.video_engine.value}")
    print(f"  HuMo:     tier={svc.config.humo.tier.value} ({svc.config.humo.model_size}) @ {svc.config.humo.resolution}")
    print(f"Image:      {svc.config.image_gen.model.value}, {svc.config.image_gen.effective_steps} steps")

    if svc.scenes.scenes:
        approved_img = sum(1 for s in svc.scenes.scenes if s.image_status == "approved")
        approved_vid = sum(1 for s in svc.scenes.scenes if s.video_status == "approved")
        print(f"  Images:   {approved_img}/{len(svc.scenes.scenes)} approved")
        print(f"  Videos:   {approved_vid}/{len(svc.scenes.scenes)} approved")


def cmd_import_audio(args: argparse.Namespace) -> None:
    """Import audio (and optionally lyrics) into a project."""
    from musicvision.project import ProjectService

    project_dir = Path(args.project).resolve()
    svc = ProjectService.open(project_dir)

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)

    dest = svc.import_audio(audio_path)
    print(f"Imported audio: {dest.name}")

    if args.lyrics:
        lyrics_path = Path(args.lyrics).resolve()
        if not lyrics_path.exists():
            print(f"Lyrics file not found: {lyrics_path}")
            sys.exit(1)
        lyrics_dest = svc.import_lyrics(lyrics_path)
        print(f"Imported lyrics: {lyrics_dest.name}")

    svc.save_config()
    print(f"Project '{svc.config.name}' updated.")


def cmd_intake(args: argparse.Namespace) -> None:
    """Run Stage 1: audio analysis + segmentation."""
    from musicvision.intake.pipeline import run_intake as _run_intake
    from musicvision.project import ProjectService

    project_dir = Path(args.project).resolve()
    svc = ProjectService.open(project_dir)

    if not svc.config.song.audio_file:
        print("No audio file in project. Run import-audio first.")
        sys.exit(1)

    device_map = None
    if not args.skip_transcription:
        try:
            from musicvision.utils.gpu import detect_devices
            device_map = detect_devices()
        except Exception:
            pass  # CPU fallback

    print(f"Running intake pipeline (llm={args.llm}, skip_transcription={args.skip_transcription})…")
    scene_list = _run_intake(
        project=svc,
        use_llm_segmentation=args.llm,
        device_map=device_map,
        skip_transcription=args.skip_transcription,
        use_vocal_separation=args.vocal_separation,
    )
    print(f"Segmented into {len(scene_list.scenes)} scenes.")
    for s in scene_list.scenes:
        print(f"  {s.id}: {s.time_start:.2f}–{s.time_end:.2f}s ({s.duration:.1f}s) {s.lyrics[:50] if s.lyrics else '(instrumental)'}…")


def cmd_generate_images(args: argparse.Namespace) -> None:
    """Generate reference images for scenes."""
    from musicvision.imaging.factory import create_engine
    from musicvision.imaging.prompt_generator import generate_image_prompt
    from musicvision.models import ImageGenConfig, ImageModel
    from musicvision.project import ProjectService
    from musicvision.utils.gpu import detect_devices

    project_dir = Path(args.project).resolve()
    svc = ProjectService.open(project_dir)

    # Override model if specified
    if args.model:
        try:
            svc.config.image_gen.model = ImageModel(args.model)
        except ValueError:
            valid = [m.value for m in ImageModel]
            print(f"Unknown model '{args.model}'. Valid: {', '.join(valid)}")
            sys.exit(1)

    # Resolve target scenes
    scenes = svc.scenes.scenes
    if args.scene_ids:
        scenes = [s for s in scenes if s.id in args.scene_ids]
        if not scenes:
            print(f"No matching scenes for IDs: {args.scene_ids}")
            sys.exit(1)

    if not scenes:
        print("No scenes. Run intake first.")
        sys.exit(1)

    # Generate prompts for scenes that don't have one yet
    for scene in scenes:
        if not scene.effective_image_prompt:
            scene.image_prompt = generate_image_prompt(scene, svc.config)

    # Resolve style sheet dimensions
    ss = svc.config.style_sheet
    res_parts = ss.resolution.split("x") if "x" in ss.resolution else ["768", "512"]
    width, height = int(res_parts[0]), int(res_parts[1])

    # Build character LoRA lookup
    char_loras: dict[str, tuple[str, float]] = {}
    for char_def in svc.config.style_sheet.characters:
        if char_def.lora_path:
            char_loras[char_def.id] = (char_def.lora_path, char_def.lora_weight)

    # Create engine and generate
    device_map = detect_devices()
    engine = create_engine(svc.config.image_gen, device_map)
    print(f"Loading image engine ({svc.config.image_gen.model.value})…")
    engine.load()

    generated = 0
    errors = []
    try:
        # Sort by LoRA to minimize swaps
        def _lora_key(s):
            for cid in s.characters:
                if cid in char_loras:
                    return char_loras[cid][0]
            return ""

        sorted_scenes = sorted(scenes, key=_lora_key)

        for scene in sorted_scenes:
            try:
                lora_path = None
                lora_weight = 0.8
                for cid in scene.characters:
                    if cid in char_loras:
                        lora_path, lora_weight = char_loras[cid]
                        break

                output_path = svc.paths.image_path(scene.id)
                prompt = scene.effective_image_prompt
                print(f"  Generating {scene.id}: {prompt[:60]}…")
                engine.generate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    lora_path=lora_path,
                    lora_weight=lora_weight,
                    output_path=output_path,
                )
                scene.reference_image = f"images/{scene.id}.png"
                generated += 1
                print(f"  {scene.id} done")
            except Exception as exc:
                print(f"  {scene.id} FAILED: {exc}")
                errors.append(scene.id)
    finally:
        engine.unload()

    svc.save_scenes()
    print(f"\nGenerated {generated}/{len(scenes)} images.")
    if errors:
        print(f"Failed: {errors}")


def cmd_assemble(args: argparse.Namespace) -> None:
    """Assemble clips into rough cut + export EDL/FCPXML."""
    from musicvision.assembly.concatenator import assemble_rough_cut
    from musicvision.assembly.exporter import export_edl, export_fcpxml
    from musicvision.project import ProjectService

    project_dir = Path(args.project).resolve()
    svc = ProjectService.open(project_dir)

    audio_file = svc.config.song.audio_file
    if not audio_file:
        print("No audio file set in project config.")
        sys.exit(1)

    audio_path = svc.resolve_path(audio_file)
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)

    print("Assembling rough cut…")
    try:
        rough_cut = assemble_rough_cut(
            scenes=svc.scenes,
            paths=svc.paths,
            original_audio=audio_path,
            approved_only=args.approved_only,
        )
    except (RuntimeError, ValueError) as e:
        print(f"Assembly failed: {e}")
        sys.exit(1)

    print(f"Rough cut: {rough_cut}")

    if not args.no_edl:
        edl = export_edl(svc.scenes, svc.paths)
        print(f"EDL:       {edl}")

    if not args.no_fcpxml:
        humo = svc.config.humo
        fcpxml = export_fcpxml(
            svc.scenes,
            svc.paths,
            width=humo.width,
            height=humo.height,
        )
        print(f"FCPXML:    {fcpxml}")

    print("Assembly complete.")


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


def cmd_upscale(args: argparse.Namespace) -> None:
    """Upscale video clips for a project."""
    from musicvision.models import TargetResolution, UpscalerType, VideoEngineType
    from musicvision.project import ProjectService
    from musicvision.upscaling.pipeline import upscale_clips

    project_dir = Path(args.project).resolve()
    svc = ProjectService.open(project_dir)

    # Override resolution if specified
    if args.resolution:
        svc.config.upscaler.target_resolution = TargetResolution(args.resolution)

    # Override upscaler if specified
    if args.upscaler:
        svc.config.upscaler.upscaler_override = UpscalerType(args.upscaler)

    render_mode = args.render_mode or "final"
    scene_ids = args.scene_ids or None

    device_map = None
    try:
        from musicvision.utils.gpu import detect_devices
        device_map = detect_devices()
    except Exception:
        pass

    target_w, target_h = svc.config.upscaler.target_width_height()
    print(f"Upscaling to {target_w}x{target_h} ({svc.config.upscaler.target_resolution.value})…")

    result = upscale_clips(
        scenes=svc.scenes,
        paths=svc.paths,
        upscaler_config=svc.config.upscaler,
        default_engine=svc.config.video_engine,
        render_mode=render_mode,
        scene_ids=scene_ids,
        device_map=device_map,
    )

    svc.save_scenes()

    upscaled = result["upscaled"]
    failed = result["failed"]
    print(f"\nUpscaled {len(upscaled)} clip(s).")
    if failed:
        print(f"Failed: {failed}")


def cmd_generate_video(args: argparse.Namespace) -> None:
    """Generate video clips for a project."""
    from musicvision.models import HumoTier, VideoEngineType
    from musicvision.project import ProjectService
    from musicvision.video.factory import create_video_engine
    from musicvision.video.prompt_generator import generate_video_prompts_batch

    project_dir = Path(args.project).resolve()
    svc = ProjectService.open(project_dir)

    # Determine engine type
    engine_type = VideoEngineType(args.engine) if args.engine else svc.config.video_engine

    # Override tier/block-swap if specified on command line (HuMo only)
    if engine_type == VideoEngineType.HUMO:
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

    # Pre-compute sub-clip frame plans
    from musicvision.engine_registry import get_constraints, plan_subclips, sub_clip_suffixes, frames_to_seconds
    constraints = get_constraints(engine_type.value)
    plan_subclips(targets, constraints, svc.paths.segments_dir, svc.paths.sub_segments_dir)
    svc.save_scenes()

    # Create engine
    if engine_type == VideoEngineType.LTX_VIDEO:
        from musicvision.utils.gpu import detect_devices
        device_map = detect_devices()
        engine = create_video_engine(svc.config.ltx_video, device_map=device_map, engine_type=engine_type)
        print(f"Loading LTX-Video 2 engine ({svc.config.ltx_video.model_id})…")
    else:
        from musicvision.utils.gpu import detect_devices
        device_map = detect_devices()
        engine = create_video_engine(svc.config.humo, device_map=device_map, engine_type=engine_type)
        print(
            f"Loading HuMo engine (tier={svc.config.humo.tier.value}, "
            f"block_swap={svc.config.humo.block_swap_count})…"
        )

    from musicvision.utils.gpu import _oom_suggestion, estimate_vram_gb, is_oom_error

    # --- Pre-flight VRAM check (advisory) ---
    if engine_type == VideoEngineType.LTX_VIDEO:
        engine_config = svc.config.ltx_video
    else:
        engine_config = svc.config.humo
    estimated = estimate_vram_gb(
        engine_type.value,
        image_size=getattr(engine_config, "image_size", 0),
        cpu_offload=getattr(engine_config, "cpu_offload", True),
    )
    if estimated > 0:
        try:
            import torch
            free_bytes, _ = torch.cuda.mem_get_info(0)
            available_gb = round(free_bytes / 1024**3, 1)
            if estimated + 2.0 > available_gb:
                print(
                    f"\n  WARNING: {engine_type.value} estimated {estimated} GB but only "
                    f"{available_gb} GB free. {_oom_suggestion(engine_type.value, engine_config)}\n"
                )
        except Exception:
            pass

    engine.load()

    generated = 0
    errors = []
    oom_scenes: list[str] = []
    consecutive_ooms = 0
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

            # Early abort: skip remaining scenes after 2 consecutive OOMs
            if consecutive_ooms >= 2:
                print(f"  SKIP {scene.id}: aborting after {consecutive_ooms} consecutive OOMs")
                oom_scenes.append(scene.id)
                continue

            prompt = scene.effective_video_prompt or scene.effective_image_prompt or scene.lyrics or f"Scene {scene.id}"
            ref_image = svc.resolve_path(scene.reference_image)
            audio_seg = svc.resolve_path(scene.audio_segment)

            # Resolve pre-computed sub-clip audio paths
            subclip_audio = None
            if scene.generation_audio_segments and len(scene.generation_audio_segments) > 1:
                from pathlib import Path as _Path
                subclip_audio = [
                    svc.resolve_path(p) if not _Path(p).is_absolute() else _Path(p)
                    for p in scene.generation_audio_segments
                ]

            print(f"  Generating {scene.id} ({scene.duration:.2f}s)…")
            try:
                outputs = engine.generate_scene(
                    text_prompt=prompt,
                    reference_image=ref_image,
                    audio_segment=audio_seg,
                    output_dir=svc.paths.clips_dir,
                    scene_id=scene.id,
                    duration=scene.duration,
                    subclip_frame_counts=scene.subclip_frame_counts,
                    subclip_audio_paths=subclip_audio,
                )
                if outputs:
                    if len(outputs) == 1:
                        scene.video_clip = str(outputs[0].video_path.relative_to(svc.paths.root))
                    else:
                        from musicvision.models import SubClip, ApprovalStatus
                        suffixes = sub_clip_suffixes(len(outputs))
                        frame_counts = scene.subclip_frame_counts or []
                        scene.sub_clips = []
                        cursor = 0
                        for i, (out, suffix) in enumerate(zip(outputs, suffixes)):
                            fc = frame_counts[i] if i < len(frame_counts) else None
                            sub_start = scene.time_start + frames_to_seconds(cursor, constraints.fps)
                            cursor += fc or 0
                            sub_end = scene.time_start + frames_to_seconds(cursor, constraints.fps)
                            sc = SubClip(
                                id=f"{scene.id}_{suffix}",
                                time_start=sub_start,
                                time_end=min(sub_end, scene.time_end),
                                video_clip=str(out.video_path.relative_to(svc.paths.root)),
                                frame_count=fc,
                            )
                            scene.sub_clips.append(sc)
                    from musicvision.models import ApprovalStatus
                    scene.video_status = ApprovalStatus.PENDING
                    generated += 1
                    consecutive_ooms = 0  # reset on success
                    print(f"  ✓ {scene.id} → {scene.video_clip or f'{len(outputs)} sub-clips'}")
            except Exception as exc:
                if is_oom_error(exc):
                    consecutive_ooms += 1
                    oom_scenes.append(scene.id)
                    print(f"  ✗ {scene.id}: OOM ({consecutive_ooms} consecutive)")
                else:
                    consecutive_ooms = 0  # non-OOM error resets counter
                    print(f"  ✗ {scene.id}: {exc}")
                    errors.append(scene.id)
    finally:
        engine.unload()

    svc.save_scenes()
    print(f"\nGenerated {generated}/{len(targets)} clips.")
    if errors:
        print(f"Failed: {errors}")
    if oom_scenes:
        suggestion = _oom_suggestion(engine_type.value, engine_config)
        print(f"\nOOM failures ({len(oom_scenes)} scenes): {oom_scenes}")
        print(f"  Suggestion: {suggestion}")
        print(f"  Re-run with adjusted settings:")
        print(f"    musicvision generate-video --project {project_dir} --scene-ids {' '.join(oom_scenes)}")


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
    p_serve = sub.add_parser("serve", help="Start the API server (optionally with a project)")
    p_serve.add_argument("directory", nargs="?", default=None, help="Project directory path (optional — omit to start empty)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port to serve on")

    # info
    p_info = sub.add_parser("info", help="Show project info")
    p_info.add_argument("directory", help="Project directory path")

    # import-audio
    p_ia = sub.add_parser("import-audio", help="Import audio (and optionally lyrics) into a project")
    p_ia.add_argument("--project", required=True, help="Project directory path")
    p_ia.add_argument("--audio", required=True, help="Path to audio file")
    p_ia.add_argument("--lyrics", default=None, help="Path to lyrics file (.txt/.lrc)")

    # intake
    p_in = sub.add_parser("intake", help="Stage 1: Analyze audio and segment into scenes")
    p_in.add_argument("--project", required=True, help="Project directory path")
    p_in.add_argument("--llm", action="store_true", help="Use LLM for segmentation (default: rule-based)")
    p_in.add_argument("--skip-transcription", action="store_true", dest="skip_transcription",
                       help="Skip Whisper transcription (use existing lyrics)")
    p_in.add_argument("--no-vocal-separation", action="store_false", dest="vocal_separation",
                       help="Skip vocal separation (runs by default)")
    p_in.set_defaults(vocal_separation=True)

    # generate-images
    p_gi = sub.add_parser("generate-images", help="Stage 2: Generate reference images for scenes")
    p_gi.add_argument("--project", required=True, help="Project directory path")
    p_gi.add_argument("--model", default=None,
                      choices=["flux-dev", "flux-schnell", "z-image", "z-image-turbo"],
                      help="Image model (default: project config)")
    p_gi.add_argument("--scene-ids", nargs="*", default=[], dest="scene_ids",
                      help="Scene IDs to generate (default: all)")

    # assemble
    p_as = sub.add_parser("assemble", help="Stage 4: Assemble clips into rough cut")
    p_as.add_argument("--project", required=True, help="Project directory path")
    p_as.add_argument("--approved-only", action="store_true", dest="approved_only",
                      help="Only include approved scenes")
    p_as.add_argument("--no-edl", action="store_true", dest="no_edl", help="Skip EDL export")
    p_as.add_argument("--no-fcpxml", action="store_true", dest="no_fcpxml", help="Skip FCPXML export")

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

    # upscale
    p_up = sub.add_parser("upscale", help="Stage 4b: Upscale video clips")
    p_up.add_argument("--project", required=True, help="Project directory path")
    p_up.add_argument(
        "--resolution", default=None,
        choices=["720p", "1080p", "1440p", "4k"],
        help="Target resolution (default: 1080p)",
    )
    p_up.add_argument(
        "--upscaler", default=None,
        choices=["ltx_spatial", "seedvr2", "real_esrgan"],
        help="Override upscaler (default: auto per engine)",
    )
    p_up.add_argument("--render-mode", default="final", choices=["preview", "final"],
                      dest="render_mode", help="Render mode (default: final)")
    p_up.add_argument("--scene-ids", nargs="*", default=[], dest="scene_ids",
                      help="Scene IDs to upscale (default: all with clips)")

    # generate-video
    p_gv = sub.add_parser("generate-video", help="Generate video clips for a project")
    p_gv.add_argument("--project", required=True, help="Project directory path")
    p_gv.add_argument(
        "--engine", default=None,
        choices=["humo", "ltx_video"],
        help="Video engine (default: project config)",
    )
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
    elif args.command == "import-audio":
        cmd_import_audio(args)
    elif args.command == "intake":
        cmd_intake(args)
    elif args.command == "generate-images":
        cmd_generate_images(args)
    elif args.command == "assemble":
        cmd_assemble(args)
    elif args.command == "detect-hardware":
        cmd_detect_hardware(args)
    elif args.command == "download-weights":
        cmd_download_weights(args)
    elif args.command == "upscale":
        cmd_upscale(args)
    elif args.command == "generate-video":
        cmd_generate_video(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
