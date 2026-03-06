"""
FastAPI application for MusicVision.

Thin HTTP layer over the core pipeline modules.
All business logic lives in ProjectService, intake/, imaging/, video/, assembly/.
This file only does request/response translation.

Environment variables are loaded from a .env file in the working directory
at import time (python-dotenv). Already-set env vars are never overwritten.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # load .env from cwd (or any parent) before anything reads env vars

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from musicvision.models import (
    ApprovalStatus,
    HumoConfig,
    ImageGenConfig,
    ImageModel,
    ProjectConfig,
    Scene,
    SceneBoundary,
    StyleSheet,
    VideoEngineType,
)

# Backward compat — keep FluxConfig importable from here
FluxConfig = ImageGenConfig
from musicvision.project import ProjectService

log = logging.getLogger(__name__)

app = FastAPI(title="MusicVision", version="0.1.0")

# CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite / CRA
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# State — single project at a time for now. Multi-project later if needed.
# ---------------------------------------------------------------------------

_project: ProjectService | None = None


def get_project() -> ProjectService:
    if _project is None:
        raise HTTPException(status_code=400, detail="No project loaded. Create or open a project first.")
    return _project


def _resolve_scene_audio(proj: ProjectService, scene, audio_path: Path) -> Path:
    """Return the audio segment for video generation.

    When lip_sync is disabled, returns a silent WAV of the same duration.
    """
    segment = proj.resolve_path(scene.audio_segment) if scene.audio_segment else audio_path
    if not scene.effective_lip_sync:
        from musicvision.utils.audio import generate_silence

        silent_path = proj.paths.segments_dir / f"{scene.id}_silent.wav"
        if not silent_path.exists():
            generate_silence(silent_path, scene.duration)
        return silent_path
    return segment


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class FilesystemListRequest(BaseModel):
    """Query parameters for filesystem listing."""
    path: str = ""
    type: str = "all"  # "directory", "file", or "all"


class ImportPathRequest(BaseModel):
    """Request body for path-based file import."""
    path: str


class CreateProjectRequest(BaseModel):
    name: str = "Untitled Project"
    directory: str  # absolute path where project will be created


class OpenProjectRequest(BaseModel):
    directory: str  # absolute path to existing project


class UpdateSceneRequest(BaseModel):
    image_prompt_user_override: Optional[str] = None
    video_prompt_user_override: Optional[str] = None
    image_status: Optional[ApprovalStatus] = None
    video_status: Optional[ApprovalStatus] = None
    lip_sync: Optional[bool] = None
    notes: Optional[str] = None


class GenerateImagesRequest(BaseModel):
    scene_ids: list[str] = []  # empty = all scenes
    model: str | None = None   # override project config model (e.g. "z-image-turbo")


class GenerateVideosRequest(BaseModel):
    scene_ids: list[str] = []
    engine: str | None = None  # override project config engine (e.g. "hunyuan_avatar")
    render_mode: str = "preview"  # "preview" (256p/10steps) or "final" (512p/30steps)


class RegenerateImageRequest(BaseModel):
    model: str | None = None   # "z-image-turbo" | "z-image" | "flux-dev" | "flux-schnell"
    seed: int = -1             # -1 = random


class RegenerateVideoRequest(BaseModel):
    engine: str | None = None  # "hunyuan_avatar" | "humo"
    seed: int = -1
    render_mode: str = "preview"  # "preview" (256p/10steps) or "final" (512p/30steps)


# ---------------------------------------------------------------------------
# Project endpoints
# ---------------------------------------------------------------------------

@app.post("/api/projects/create")
async def create_project(req: CreateProjectRequest):
    global _project
    project_dir = Path(req.directory).resolve()
    _project = ProjectService.create(project_dir, name=req.name)
    mount_project_files(project_dir)
    return {"status": "created", "name": req.name, "directory": req.directory}


@app.post("/api/projects/open")
async def open_project(req: OpenProjectRequest):
    global _project
    try:
        project_dir = Path(req.directory).resolve()
        _project = ProjectService.open(project_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    mount_project_files(project_dir)
    return {"status": "opened", "name": _project.config.name, "directory": req.directory}


@app.get("/api/projects/config")
async def get_config() -> ProjectConfig:
    return get_project().config


@app.put("/api/projects/config")
async def update_config(config: ProjectConfig):
    proj = get_project()
    proj.config = config
    proj.save_config()
    return {"status": "updated"}


@app.put("/api/projects/config/style-sheet")
async def update_style_sheet(style_sheet: StyleSheet):
    proj = get_project()
    proj.config.style_sheet = style_sheet
    proj.save_config()
    return {"status": "updated"}


@app.put("/api/projects/config/humo")
async def update_humo_config(humo: HumoConfig):
    proj = get_project()
    proj.config.humo = humo
    proj.save_config()
    return {"status": "updated"}


@app.put("/api/projects/config/image-gen")
async def update_image_gen_config(image_gen: ImageGenConfig):
    proj = get_project()
    proj.config.image_gen = image_gen
    proj.save_config()
    return {"status": "updated"}


@app.put("/api/projects/config/flux")
async def update_flux_config(flux: ImageGenConfig):
    """Deprecated — use /api/projects/config/image-gen instead."""
    proj = get_project()
    proj.config.image_gen = flux
    proj.save_config()
    return {"status": "updated"}


# ---------------------------------------------------------------------------
# File upload endpoints
# ---------------------------------------------------------------------------

@app.post("/api/upload/audio")
async def upload_audio(file: UploadFile):
    proj = get_project()
    dest = proj.paths.input_dir / file.filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    proj.config.song.audio_file = f"input/{file.filename}"
    proj.save_config()
    return {"status": "uploaded", "path": str(dest)}


@app.post("/api/upload/lyrics")
async def upload_lyrics(file: UploadFile):
    proj = get_project()
    dest = proj.paths.input_dir / file.filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    proj.config.song.lyrics_file = f"input/{file.filename}"
    proj.save_config()
    return {"status": "uploaded", "path": str(dest)}


@app.post("/api/upload/acestep-json")
async def upload_acestep_json(file: UploadFile):
    """Upload an AceStep metadata JSON separately from the audio."""
    proj = get_project()
    dest = proj.paths.input_dir / file.filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    proj.import_acestep_json(dest)
    return {
        "status": "uploaded",
        "bpm": proj.config.song.bpm,
        "duration": proj.config.song.duration_seconds,
        "keyscale": proj.config.song.keyscale,
        "has_lyrics": bool(proj.config.song.lyrics_file),
    }


# ---------------------------------------------------------------------------
# Filesystem browser endpoints
# ---------------------------------------------------------------------------

@app.get("/api/filesystem/list")
async def list_filesystem(path: str = "", type: str = "all"):
    """Browse the local filesystem for files and directories.

    Args:
        path: Directory to list. Defaults to home directory.
        type: Filter entries — "directory", "file", or "all" (default).
    """
    if not path:
        target = Path.home()
    else:
        target = Path(path).expanduser().resolve()

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {target}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {target}")

    entries = []
    try:
        for item in target.iterdir():
            # Skip hidden files/directories
            if item.name.startswith("."):
                continue
            try:
                is_dir = item.is_dir()
            except PermissionError:
                continue
            if type == "directory" and not is_dir:
                continue
            if type == "file" and is_dir:
                continue
            entries.append({
                "name": item.name,
                "path": str(item),
                "is_dir": is_dir,
            })
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {target}")

    # Sort: directories first, then files, alphabetical within each group
    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))

    parent = str(target.parent) if target.parent != target else None
    return {"entries": entries, "parent": parent}


# ---------------------------------------------------------------------------
# Path-based import endpoints
# ---------------------------------------------------------------------------

@app.post("/api/import/audio")
async def import_audio(req: ImportPathRequest):
    """Import audio file from a local path. Auto-detects sibling AceStep JSON."""
    proj = get_project()
    source = Path(req.path).expanduser().resolve()
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {source}")
    if not source.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {source}")

    # Check for companion AceStep JSON before import (to report in response)
    acestep_json = source.with_suffix(".json")
    has_acestep = acestep_json.exists()

    proj.import_audio(source)

    result: dict = {
        "status": "imported",
        "path": str(source),
        "acestep_imported": has_acestep,
    }
    if has_acestep:
        result["bpm"] = proj.config.song.bpm
        result["duration_seconds"] = proj.config.song.duration_seconds
        result["keyscale"] = proj.config.song.keyscale
        result["has_lyrics"] = bool(proj.config.song.lyrics_file)
    return result


@app.post("/api/import/lyrics")
async def import_lyrics(req: ImportPathRequest):
    """Import lyrics file from a local path."""
    proj = get_project()
    source = Path(req.path).expanduser().resolve()
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {source}")
    if not source.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {source}")

    proj.import_lyrics(source)
    return {"status": "imported", "path": str(source)}


# ---------------------------------------------------------------------------
# Scene endpoints
# ---------------------------------------------------------------------------

@app.get("/api/scenes")
async def list_scenes() -> list[Scene]:
    return get_project().scenes.scenes


@app.get("/api/scenes/{scene_id}")
async def get_scene(scene_id: str) -> Scene:
    scene = get_project().scenes.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
    return scene


@app.patch("/api/scenes/{scene_id}")
async def update_scene(scene_id: str, req: UpdateSceneRequest):
    proj = get_project()
    scene = proj.scenes.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    if req.image_prompt_user_override is not None:
        scene.image_prompt_user_override = req.image_prompt_user_override
    if req.video_prompt_user_override is not None:
        scene.video_prompt_user_override = req.video_prompt_user_override
    if req.image_status is not None:
        scene.image_status = req.image_status
    if req.video_status is not None:
        scene.video_status = req.video_status
    if req.lip_sync is not None:
        scene.lip_sync = req.lip_sync
    if req.notes is not None:
        scene.notes = req.notes

    proj.save_scenes()
    return scene


@app.post("/api/scenes/approve-all")
async def approve_all_scenes():
    proj = get_project()
    for scene in proj.scenes.scenes:
        if scene.reference_image:
            scene.image_status = ApprovalStatus.APPROVED
        if scene.video_clip:
            scene.video_status = ApprovalStatus.APPROVED
    proj.save_scenes()
    return {"status": "approved", "count": len(proj.scenes.scenes)}


# ---------------------------------------------------------------------------
# Per-scene regeneration endpoints
# ---------------------------------------------------------------------------

@app.post("/api/scenes/{scene_id}/regenerate-image")
async def regenerate_image(scene_id: str, req: RegenerateImageRequest) -> Scene:
    """Generate or regenerate a single scene's reference image."""
    from musicvision.imaging import create_engine
    from musicvision.imaging.prompt_generator import generate_image_prompt
    from musicvision.utils.gpu import detect_devices

    proj = get_project()
    scene = proj.scenes.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    # Auto-generate prompt if missing
    prompt = scene.effective_image_prompt
    if not prompt:
        scene.image_prompt = generate_image_prompt(scene, proj.config)
        prompt = scene.effective_image_prompt
        if not prompt:
            raise HTTPException(status_code=400, detail="Failed to generate image prompt.")

    # Build config — override model if provided
    config = proj.config.image_gen.model_copy()
    if req.model:
        config.model = ImageModel(req.model)

    # Resolve dimensions from style sheet
    ss = proj.config.style_sheet
    res_parts = ss.resolution.split("x") if "x" in ss.resolution else ["1280", "720"]
    width, height = int(res_parts[0]), int(res_parts[1])

    # Character LoRA
    char_loras: dict[str, tuple[str, float]] = {}
    for char_def in proj.config.style_sheet.characters:
        if char_def.lora_path:
            char_loras[char_def.id] = (char_def.lora_path, char_def.lora_weight)

    lora_path = None
    lora_weight = 0.8
    for cid in scene.characters:
        if cid in char_loras:
            lora_path, lora_weight = char_loras[cid]
            break

    import asyncio

    def _run() -> Scene:
        device_map = detect_devices()
        engine = create_engine(config, device_map)
        engine.load()
        try:
            output_path = proj.paths.image_path(scene.id)
            seed = req.seed if req.seed >= 0 else None
            engine.generate(
                prompt=prompt,
                width=width,
                height=height,
                lora_path=lora_path,
                lora_weight=lora_weight,
                output_path=output_path,
                seed=seed,
            )
            scene.reference_image = f"images/{scene.id}.png"
            scene.image_status = ApprovalStatus.PENDING
        finally:
            engine.unload()
        proj.save_scenes()
        return scene

    return await asyncio.to_thread(_run)


@app.post("/api/scenes/{scene_id}/regenerate-video")
async def regenerate_video(scene_id: str, req: RegenerateVideoRequest) -> Scene:
    """Generate or regenerate a single scene's video clip at preview quality."""
    from musicvision.engine_registry import (
        frames_to_seconds,
        get_constraints,
        plan_subclips,
        sub_clip_suffixes,
    )
    from musicvision.models import SubClip
    from musicvision.video import create_video_engine
    from musicvision.video.prompt_generator import generate_video_prompt

    proj = get_project()
    scene = proj.scenes.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    # Auto-generate prompt if missing
    prompt = scene.effective_video_prompt
    if not prompt:
        scene.video_prompt = generate_video_prompt(scene, proj.config)
        prompt = scene.effective_video_prompt
        if not prompt:
            raise HTTPException(status_code=400, detail="Failed to generate video prompt.")
    if not scene.reference_image:
        raise HTTPException(status_code=400, detail="Scene has no reference image. Generate one first.")

    # Resolve audio (silence if lip_sync is off)
    audio_file = proj.config.song.audio_file
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file in project config.")
    audio_path = proj.resolve_path(audio_file)
    segment = _resolve_scene_audio(proj, scene, audio_path)

    # Determine engine type
    engine_type = VideoEngineType(req.engine) if req.engine else (scene.video_engine or proj.config.video_engine)

    # Pre-compute sub-clip frame plan
    constraints = get_constraints(engine_type.value)
    plan_subclips([scene], constraints, proj.paths.segments_dir, proj.paths.sub_segments_dir)

    # Apply render mode
    if engine_type == VideoEngineType.HUNYUAN_AVATAR:
        config = proj.config.hunyuan_avatar.model_copy()
        if req.render_mode == "preview":
            config.image_size = 256
            config.infer_steps = 10
        else:
            config.image_size = 512
            config.infer_steps = 30
        engine = create_video_engine(config, engine_type=engine_type)
    elif engine_type == VideoEngineType.LTX_VIDEO:
        from musicvision.utils.gpu import detect_devices

        config = proj.config.ltx_video.model_copy()
        if req.render_mode == "preview":
            config.width = 480
            config.height = 320
            config.num_inference_steps = 20
        else:
            config.width = 768
            config.height = 512
            config.num_inference_steps = 40
        device_map = detect_devices()
        engine = create_video_engine(config, device_map=device_map, engine_type=engine_type)
    else:
        from musicvision.utils.gpu import detect_devices

        config = proj.config.humo.model_copy()
        device_map = detect_devices()
        engine = create_video_engine(config, device_map=device_map, engine_type=engine_type)

    # Resolve seed: use locked seed if approved, use request seed if specified, else random
    import random
    if scene.video_status.value == "approved" and scene.video_seed is not None:
        scene_seed = scene.video_seed
    elif req.seed and req.seed >= 0:
        scene_seed = req.seed
    else:
        scene_seed = random.randint(0, 2**31 - 1)
    scene.video_seed = scene_seed
    config.seed = scene_seed

    # Resolve pre-computed sub-clip audio paths
    subclip_audio = None
    if scene.generation_audio_segments and len(scene.generation_audio_segments) > 1:
        subclip_audio = [
            proj.resolve_path(p) if not Path(p).is_absolute() else Path(p)
            for p in scene.generation_audio_segments
        ]

    import asyncio

    def _run() -> Scene:
        engine.load()
        try:
            ref_image = proj.resolve_path(scene.reference_image)
            results = engine.generate_scene(
                text_prompt=prompt,
                reference_image=ref_image,
                audio_segment=segment,
                output_dir=proj.paths.clips_dir,
                scene_id=scene.id,
                duration=scene.duration,
                subclip_frame_counts=scene.subclip_frame_counts,
                subclip_audio_paths=subclip_audio,
            )

            if len(results) > 1:
                scene.sub_clips = []
                suffixes = sub_clip_suffixes(len(results))
                frame_counts = scene.subclip_frame_counts or []
                cursor = 0
                for j, (r, suffix) in enumerate(zip(results, suffixes)):
                    fc = frame_counts[j] if j < len(frame_counts) else None
                    sub_start = scene.time_start + frames_to_seconds(cursor, constraints.fps)
                    cursor += fc or 0
                    sub_end = scene.time_start + frames_to_seconds(cursor, constraints.fps)
                    scene.sub_clips.append(SubClip(
                        id=f"{scene.id}_{suffix}",
                        time_start=sub_start,
                        time_end=min(sub_end, scene.time_end),
                        video_prompt=prompt,
                        video_clip=str(r.video_path.relative_to(proj.paths.root)),
                        frame_count=fc,
                    ))
                from musicvision.utils.audio import concat_videos

                sub_paths = [r.video_path for r in results]
                joined = proj.paths.clips_dir / f"{scene.id}_joined.mp4"
                concat_videos(sub_paths, joined)
                scene.video_clip = str(joined.relative_to(proj.paths.root))
            else:
                scene.video_clip = f"clips/{scene.id}.mp4"
                scene.sub_clips = []

            scene.video_status = ApprovalStatus.PENDING
            from musicvision.utils.video import update_scene_resolution
            update_scene_resolution(scene, proj.paths.root)
        finally:
            engine.unload()

        proj.save_scenes()
        return scene

    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Pipeline stage endpoints (stubs — will call into engine modules)
# ---------------------------------------------------------------------------

@app.post("/api/pipeline/analyze")
async def analyze_audio(
    skip_transcription: bool = False,
    use_vocal_separation: bool = True,
):
    """Phase 1: Analyze audio (BPM, Whisper, demucs). No scene boundaries."""
    from musicvision.intake.pipeline import run_analyze as _run_analyze
    from musicvision.utils.gpu import detect_devices

    proj = get_project()

    def _run() -> dict:
        device_map = detect_devices()
        result = _run_analyze(
            project=proj,
            device_map=device_map,
            skip_transcription=skip_transcription,
            use_vocal_separation=use_vocal_separation,
        )
        return result.model_dump(mode="json")

    return await asyncio.to_thread(_run)


class CreateScenesRequest(BaseModel):
    boundaries: list[SceneBoundary]
    snap_to_beats: bool = False


@app.post("/api/pipeline/create-scenes")
async def create_scenes(req: CreateScenesRequest):
    """Phase 2: Create scenes from manual waveform editor boundaries."""
    from musicvision.intake.pipeline import create_scenes_from_boundaries

    proj = get_project()

    def _run() -> dict:
        scene_list = create_scenes_from_boundaries(
            project=proj,
            boundaries=req.boundaries,
            snap_to_beats=req.snap_to_beats,
        )
        return {"status": "complete", "scene_count": len(scene_list.scenes)}

    return await asyncio.to_thread(_run)


@app.post("/api/pipeline/auto-segment")
async def auto_segment(use_llm: bool = True):
    """Phase 2 alternative: LLM/rule-based auto-segmentation."""
    from musicvision.intake.pipeline import run_auto_segment

    proj = get_project()

    def _run() -> dict:
        scene_list = run_auto_segment(project=proj, use_llm=use_llm)
        return {"status": "complete", "scene_count": len(scene_list.scenes)}

    return await asyncio.to_thread(_run)


@app.get("/api/analysis")
async def get_analysis():
    """Retrieve stored analysis data for the waveform editor."""
    import json as _json

    proj = get_project()
    config = proj.config

    if not config.song.analyzed:
        return {"analyzed": False}

    # Load word timestamps from disk
    ts_path = proj.paths.input_dir / "word_timestamps.json"
    word_timestamps = []
    if ts_path.exists():
        word_timestamps = _json.loads(ts_path.read_text(encoding="utf-8"))

    vocal_path = None
    vocal_file = proj.paths.input_dir / "audio_vocal.wav"
    if vocal_file.exists():
        vocal_path = str(vocal_file.relative_to(proj.paths.root))

    return {
        "analyzed": True,
        "duration": config.song.duration_seconds,
        "bpm": config.song.bpm,
        "beat_times": config.song.beat_times,
        "word_timestamps": word_timestamps,
        "vocal_path": vocal_path,
        "sections": [s.model_dump() for s in config.song.sections],
    }


@app.post("/api/pipeline/intake")
async def run_intake(
    use_llm: bool = True,
    skip_transcription: bool = False,
    use_vocal_separation: bool = True,
):
    """Stage 1: Full intake (analyze + segment in one call). CLI backward-compat."""
    from musicvision.intake.pipeline import run_intake as _run_intake
    from musicvision.utils.gpu import detect_devices

    proj = get_project()

    def _run() -> dict:
        device_map = detect_devices()
        scene_list = _run_intake(
            project=proj,
            use_llm_segmentation=use_llm,
            device_map=device_map,
            skip_transcription=skip_transcription,
            use_vocal_separation=use_vocal_separation,
        )
        return {"status": "complete", "scene_count": len(scene_list.scenes)}

    return await asyncio.to_thread(_run)


@app.post("/api/pipeline/generate-images")
async def generate_images(req: GenerateImagesRequest):
    """Stage 2: Generate reference images for specified scenes (or all)."""
    from musicvision.imaging import create_engine
    from musicvision.imaging.prompt_generator import generate_image_prompt
    from musicvision.utils.gpu import detect_devices

    proj = get_project()

    # Apply model override if provided
    if req.model:
        proj.config.image_gen.model = ImageModel(req.model)
        proj.save_config()

    # Resolve target scenes
    if req.scene_ids:
        scenes = []
        for sid in req.scene_ids:
            scene = proj.scenes.get_scene(sid)
            if not scene:
                raise HTTPException(status_code=404, detail=f"Scene {sid} not found")
            scenes.append(scene)
    else:
        scenes = proj.scenes.scenes

    if not scenes:
        raise HTTPException(status_code=400, detail="No scenes to process")

    # Generate prompts for scenes that don't have one yet
    for scene in scenes:
        if not scene.effective_image_prompt:
            scene.image_prompt = generate_image_prompt(scene, proj.config)

    # Resolve style sheet dimensions
    ss = proj.config.style_sheet
    res_parts = ss.resolution.split("x") if "x" in ss.resolution else ["1280", "720"]
    width, height = int(res_parts[0]), int(res_parts[1])

    # Build character LoRA lookup
    char_loras: dict[str, tuple[str, float]] = {}
    for char_def in proj.config.style_sheet.characters:
        if char_def.lora_path:
            char_loras[char_def.id] = (char_def.lora_path, char_def.lora_weight)

    # Run generation in a thread so the event loop stays responsive
    import asyncio

    def _run_generation() -> dict:
        device_map = detect_devices()
        engine = create_engine(proj.config.image_gen, device_map)
        engine.load()

        generated: list[str] = []
        failed: list[dict] = []

        try:
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

                    output_path = proj.paths.image_path(scene.id)
                    engine.generate(
                        prompt=scene.effective_image_prompt,
                        width=width,
                        height=height,
                        lora_path=lora_path,
                        lora_weight=lora_weight,
                        output_path=output_path,
                    )
                    scene.reference_image = f"images/{scene.id}.png"
                    generated.append(scene.id)
                    proj.save_scenes()
                except Exception as exc:
                    log.error("Image generation failed for %s: %s", scene.id, exc)
                    failed.append({"scene_id": scene.id, "error": str(exc)})
        finally:
            engine.unload()

        return {
            "status": "complete",
            "generated": generated,
            "failed": failed,
            "total": len(scenes),
        }

    return await asyncio.to_thread(_run_generation)


@app.post("/api/pipeline/generate-videos")
async def generate_videos(req: GenerateVideosRequest):
    """Stage 3: Generate video clips for specified scenes (or all).

    Scenes are grouped by their ``video_engine`` field (falling back to the
    project default).  Each engine group is processed sequentially so only one
    engine holds GPU memory at a time.
    """
    from collections import defaultdict

    from musicvision.engine_registry import (
        frames_to_seconds,
        get_constraints,
        plan_subclips,
        sub_clip_suffixes,
    )
    from musicvision.models import SubClip, VideoEngineType
    from musicvision.video import create_video_engine
    from musicvision.video.prompt_generator import generate_video_prompt

    proj = get_project()

    # Engine override for this run only (don't persist to project.yaml)
    run_engine = VideoEngineType(req.engine) if req.engine else proj.config.video_engine

    # Resolve target scenes
    if req.scene_ids:
        scenes = []
        for sid in req.scene_ids:
            scene = proj.scenes.get_scene(sid)
            if not scene:
                raise HTTPException(status_code=404, detail=f"Scene {sid} not found")
            scenes.append(scene)
    else:
        scenes = proj.scenes.scenes

    if not scenes:
        raise HTTPException(status_code=400, detail="No scenes to process")

    # Check that scenes have reference images (Stage 2 must run first)
    missing_images = [s.id for s in scenes if not s.reference_image]
    if missing_images:
        raise HTTPException(
            status_code=400,
            detail=f"Scenes missing reference images (run generate-images first): {missing_images}",
        )

    # Generate video prompts for scenes that don't have one yet
    for scene in scenes:
        if not scene.effective_video_prompt:
            scene.video_prompt = generate_video_prompt(scene, proj.config)

    # Resolve audio file
    audio_file = proj.config.song.audio_file
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file set in project config.")
    audio_path = proj.resolve_path(audio_file)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")

    # Group scenes by engine type
    engine_groups: dict[VideoEngineType, list] = defaultdict(list)
    for scene in sorted(scenes, key=lambda s: s.order):
        etype = scene.video_engine or run_engine
        engine_groups[etype].append(scene)

    from musicvision.utils.gpu import _oom_suggestion, estimate_vram_gb, is_oom_error

    generated: list[str] = []
    failed: list[dict] = []
    vram_warnings: list[dict] = []

    # --- Pre-flight VRAM check (advisory, does not block) ---
    for engine_type, group_scenes in engine_groups.items():
        estimated = estimate_vram_gb(
            engine_type.value,
            image_size=getattr(
                proj.config.hunyuan_avatar if engine_type == VideoEngineType.HUNYUAN_AVATAR else proj.config.humo,
                "image_size", 0,
            ),
            cpu_offload=getattr(
                proj.config.hunyuan_avatar if engine_type == VideoEngineType.HUNYUAN_AVATAR else proj.config.humo,
                "cpu_offload", True,
            ),
        )
        if estimated > 0:
            try:
                import torch
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                available_gb = round(free_bytes / 1024**3, 1)
            except Exception:
                available_gb = None

            if available_gb is not None and estimated + 2.0 > available_gb:
                engine_config = (
                    proj.config.hunyuan_avatar
                    if engine_type == VideoEngineType.HUNYUAN_AVATAR
                    else proj.config.humo
                )
                warning = {
                    "engine": engine_type.value,
                    "estimated_gb": estimated,
                    "available_gb": available_gb,
                    "message": (
                        f"{engine_type.value} estimated {estimated} GB but only "
                        f"{available_gb} GB available. "
                        f"{_oom_suggestion(engine_type.value, engine_config)}"
                    ),
                }
                vram_warnings.append(warning)
                log.warning("VRAM pre-flight: %s", warning["message"])

    # --- Apply render mode to engine configs (ephemeral, not saved to disk) ---
    render_mode = req.render_mode
    if render_mode == "preview":
        proj.config.hunyuan_avatar.image_size = 256
        proj.config.hunyuan_avatar.infer_steps = 10
        proj.config.ltx_video.width = 480
        proj.config.ltx_video.height = 320
        proj.config.ltx_video.num_inference_steps = 20
    else:  # "final"
        proj.config.hunyuan_avatar.image_size = 512
        proj.config.hunyuan_avatar.infer_steps = 30
        proj.config.ltx_video.width = 768
        proj.config.ltx_video.height = 512
        proj.config.ltx_video.num_inference_steps = 40
    log.info("Render mode: %s", render_mode)

    # --- Run generation in a thread so the event loop stays responsive ---
    import asyncio

    def _run_generation() -> dict:
        generated: list[str] = []
        failed: list[dict] = []

        for engine_type, group_scenes in engine_groups.items():
            constraints = get_constraints(engine_type.value)
            plan_subclips(group_scenes, constraints, proj.paths.segments_dir, proj.paths.sub_segments_dir)

            if engine_type == VideoEngineType.HUNYUAN_AVATAR:
                engine = create_video_engine(proj.config.hunyuan_avatar, engine_type=engine_type)
            elif engine_type == VideoEngineType.LTX_VIDEO:
                from musicvision.utils.gpu import detect_devices
                device_map = detect_devices()
                engine = create_video_engine(proj.config.ltx_video, device_map=device_map, engine_type=engine_type)
            else:
                from musicvision.utils.gpu import detect_devices
                device_map = detect_devices()
                engine = create_video_engine(proj.config.humo, device_map=device_map, engine_type=engine_type)

            engine.load()
            consecutive_ooms = 0

            try:
                for scene in group_scenes:
                    if consecutive_ooms >= 2:
                        engine_config = (
                            proj.config.hunyuan_avatar
                            if engine_type == VideoEngineType.HUNYUAN_AVATAR
                            else proj.config.humo
                        )
                        log.warning(
                            "Skipping %s — %d consecutive OOMs, remaining scenes in "
                            "%s group will also fail",
                            scene.id, consecutive_ooms, engine_type.value,
                        )
                        failed.append({
                            "scene_id": scene.id,
                            "error": f"Skipped after {consecutive_ooms} consecutive OOMs",
                            "error_type": "oom_skipped",
                            "oom_context": {
                                "engine": engine_type.value,
                                "image_size": getattr(engine_config, "image_size", 0),
                                "suggestion": _oom_suggestion(engine_type.value, engine_config),
                            },
                        })
                        continue

                    try:
                        import random
                        if scene.video_status.value == "approved" and scene.video_seed is not None:
                            scene_seed = scene.video_seed
                        else:
                            scene_seed = random.randint(0, 2**31 - 1)
                            scene.video_seed = scene_seed

                        if engine_type == VideoEngineType.HUNYUAN_AVATAR:
                            proj.config.hunyuan_avatar.seed = scene_seed
                        elif engine_type == VideoEngineType.LTX_VIDEO:
                            proj.config.ltx_video.seed = scene_seed
                        else:
                            proj.config.humo.seed = scene_seed

                        ref_image = proj.resolve_path(scene.reference_image)
                        segment = _resolve_scene_audio(proj, scene, audio_path)

                        subclip_audio = None
                        if scene.generation_audio_segments and len(scene.generation_audio_segments) > 1:
                            subclip_audio = [
                                proj.resolve_path(p) if not Path(p).is_absolute() else Path(p)
                                for p in scene.generation_audio_segments
                            ]

                        results = engine.generate_scene(
                            text_prompt=scene.effective_video_prompt,
                            reference_image=ref_image,
                            audio_segment=segment,
                            output_dir=proj.paths.clips_dir,
                            scene_id=scene.id,
                            duration=scene.duration,
                            subclip_frame_counts=scene.subclip_frame_counts,
                            subclip_audio_paths=subclip_audio,
                        )

                        if len(results) > 1:
                            scene.sub_clips = []
                            suffixes = sub_clip_suffixes(len(results))
                            frame_counts = scene.subclip_frame_counts or []
                            cursor = 0
                            for j, (r, suffix) in enumerate(zip(results, suffixes)):
                                fc = frame_counts[j] if j < len(frame_counts) else None
                                sub_start = scene.time_start + frames_to_seconds(cursor, constraints.fps)
                                cursor += fc or 0
                                sub_end = scene.time_start + frames_to_seconds(cursor, constraints.fps)
                                scene.sub_clips.append(SubClip(
                                    id=f"{scene.id}_{suffix}",
                                    time_start=sub_start,
                                    time_end=min(sub_end, scene.time_end),
                                    video_prompt=scene.effective_video_prompt,
                                    video_clip=str(r.video_path.relative_to(proj.paths.root)),
                                    frame_count=fc,
                                ))
                            from musicvision.utils.audio import concat_videos

                            sub_paths = [r.video_path for r in results]
                            joined = proj.paths.clips_dir / f"{scene.id}_joined.mp4"
                            concat_videos(sub_paths, joined)
                            scene.video_clip = str(joined.relative_to(proj.paths.root))
                        else:
                            scene.video_clip = f"clips/{scene.id}.mp4"

                        from musicvision.utils.video import update_scene_resolution
                        update_scene_resolution(scene, proj.paths.root)
                        generated.append(scene.id)
                        consecutive_ooms = 0
                        proj.save_scenes()

                    except Exception as exc:
                        if is_oom_error(exc):
                            consecutive_ooms += 1
                            engine_config = (
                                proj.config.hunyuan_avatar
                                if engine_type == VideoEngineType.HUNYUAN_AVATAR
                                else proj.config.humo
                            )
                            log.error(
                                "OOM on %s (%d consecutive): %s", scene.id, consecutive_ooms, exc,
                            )
                            failed.append({
                                "scene_id": scene.id,
                                "error": str(exc),
                                "error_type": "oom",
                                "oom_context": {
                                    "engine": engine_type.value,
                                    "image_size": getattr(engine_config, "image_size", 0),
                                    "suggestion": _oom_suggestion(engine_type.value, engine_config),
                                },
                            })
                        else:
                            consecutive_ooms = 0
                            log.error("Video generation failed for %s: %s", scene.id, exc)
                            failed.append({"scene_id": scene.id, "error": str(exc)})
            finally:
                engine.unload()

        result: dict = {
            "status": "complete",
            "generated": generated,
            "failed": failed,
            "total": len(scenes),
        }
        if vram_warnings:
            result["vram_warnings"] = vram_warnings
        return result

    return await asyncio.to_thread(_run_generation)


class UpscaleRequest(BaseModel):
    scene_ids: list[str] = []
    resolution: str | None = None       # "720p" | "1080p" | "1440p" | "4k"
    upscaler: str | None = None         # override: "ltx_spatial" | "seedvr2" | "real_esrgan"
    render_mode: str = "final"          # "preview" | "final"


class AssembleRequest(BaseModel):
    approved_only: bool = False
    export_edl: bool = True
    export_fcpxml: bool = True


@app.post("/api/pipeline/upscale")
async def upscale_videos(req: UpscaleRequest):
    """Stage 4b: Upscale video clips to target resolution."""
    from musicvision.models import TargetResolution, UpscalerType
    from musicvision.upscaling.pipeline import upscale_clips

    import asyncio

    proj = get_project()

    # Apply overrides (ephemeral, not saved to disk)
    if req.resolution:
        proj.config.upscaler.target_resolution = TargetResolution(req.resolution)
    if req.upscaler:
        proj.config.upscaler.upscaler_override = UpscalerType(req.upscaler)

    scene_ids = req.scene_ids or None

    def _run() -> dict:
        device_map = None
        try:
            from musicvision.utils.gpu import detect_devices
            device_map = detect_devices()
        except Exception:
            pass

        result = upscale_clips(
            scenes=proj.scenes,
            paths=proj.paths,
            upscaler_config=proj.config.upscaler,
            default_engine=proj.config.video_engine,
            render_mode=req.render_mode,
            scene_ids=scene_ids,
            device_map=device_map,
        )
        proj.save_scenes()
        return {
            "status": "complete",
            "upscaled": result["upscaled"],
            "failed": result["failed"],
            "total": len(result["upscaled"]) + len(result["failed"]),
        }

    return await asyncio.to_thread(_run)


@app.post("/api/scenes/{scene_id}/upscale")
async def upscale_scene(scene_id: str, req: UpscaleRequest):
    """Upscale a single scene's video clip."""
    from musicvision.models import TargetResolution, UpscalerType
    from musicvision.upscaling.pipeline import upscale_clips

    import asyncio

    proj = get_project()
    scene = proj.scenes.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")
    if not scene.video_clip and not scene.sub_clips:
        raise HTTPException(status_code=400, detail="Scene has no video clip to upscale")

    if req.resolution:
        proj.config.upscaler.target_resolution = TargetResolution(req.resolution)
    if req.upscaler:
        proj.config.upscaler.upscaler_override = UpscalerType(req.upscaler)

    def _run() -> dict:
        device_map = None
        try:
            from musicvision.utils.gpu import detect_devices
            device_map = detect_devices()
        except Exception:
            pass

        result = upscale_clips(
            scenes=proj.scenes,
            paths=proj.paths,
            upscaler_config=proj.config.upscaler,
            default_engine=proj.config.video_engine,
            render_mode=req.render_mode,
            scene_ids=[scene_id],
            device_map=device_map,
        )
        proj.save_scenes()
        return {
            "status": "complete",
            "upscaled": result["upscaled"],
            "failed": result["failed"],
        }

    return await asyncio.to_thread(_run)


@app.post("/api/pipeline/assemble")
async def assemble(req: AssembleRequest = AssembleRequest()):
    """Stage 4: Concatenate clips, sync audio, export EDL/FCPXML."""
    from musicvision.assembly.concatenator import assemble_rough_cut
    from musicvision.assembly.exporter import export_edl, export_fcpxml
    from musicvision.utils.audio import get_duration

    proj = get_project()
    audio_file = proj.config.song.audio_file
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file set in project config.")

    audio_path = proj.resolve_path(audio_file)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")

    def _run() -> dict:
        try:
            rough_cut = assemble_rough_cut(
                scenes=proj.scenes,
                paths=proj.paths,
                original_audio=audio_path,
                approved_only=req.approved_only,
            )
        except (RuntimeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

        scene_dur = proj.scenes.total_duration
        video_dur = get_duration(rough_cut)

        result: dict = {
            "status": "complete",
            "rough_cut": str(rough_cut.relative_to(proj.paths.root)),
            "clip_count": len([s for s in proj.scenes.scenes if s.video_clip or s.sub_clips]),
            "total_scenes": len(proj.scenes.scenes),
            "duration_seconds": scene_dur,
            "video_duration_seconds": video_dur,
            "drift_seconds": round(video_dur - scene_dur, 3),
            "output_dir": str(proj.paths.root / "output"),
        }

        if req.export_edl:
            edl = export_edl(proj.scenes, proj.paths)
            result["edl"] = str(edl.relative_to(proj.paths.root))

        if req.export_fcpxml:
            humo = proj.config.humo
            fcpxml = export_fcpxml(
                proj.scenes,
                proj.paths,
                width=humo.width,
                height=humo.height,
            )
            result["fcpxml"] = str(fcpxml.relative_to(proj.paths.root))

        return result

    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Static file serving for generated assets
# ---------------------------------------------------------------------------

def mount_project_files(project_dir: Path) -> None:
    """Mount the project directory as static files so the frontend can load images/clips."""
    # Remove existing mount if present (allows re-mounting when opening a different project)
    app.routes[:] = [r for r in app.routes if getattr(r, "name", None) != "project_files"]
    app.mount("/files", StaticFiles(directory=str(project_dir)), name="project_files")
