"""
FastAPI application for MusicVision.

Thin HTTP layer over the core pipeline modules.
All business logic lives in ProjectService, intake/, imaging/, video/, assembly/.
This file only does request/response translation.

Environment variables are loaded from a .env file in the working directory
at import time (python-dotenv). Already-set env vars are never overwritten.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

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
    notes: Optional[str] = None


class GenerateRequest(BaseModel):
    scene_ids: list[str] = []  # empty = all scenes


class RegenerateImageRequest(BaseModel):
    model: str | None = None   # "z-image-turbo" | "z-image" | "flux-dev" | "flux-schnell"
    seed: int = -1             # -1 = random


class RegenerateVideoRequest(BaseModel):
    engine: str | None = None  # "hunyuan_avatar" | "humo"
    seed: int = -1


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
    """Regenerate a single scene's reference image."""
    from musicvision.imaging import create_engine
    from musicvision.utils.gpu import detect_devices

    proj = get_project()
    scene = proj.scenes.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    prompt = scene.effective_image_prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="Scene has no image prompt. Set one first.")

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


@app.post("/api/scenes/{scene_id}/regenerate-video")
async def regenerate_video(scene_id: str, req: RegenerateVideoRequest) -> Scene:
    """Regenerate a single scene's video clip at preview quality."""
    from musicvision.engine_registry import (
        frames_to_seconds,
        get_constraints,
        plan_subclips,
        sub_clip_suffixes,
    )
    from musicvision.models import SubClip
    from musicvision.video import create_video_engine

    proj = get_project()
    scene = proj.scenes.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    prompt = scene.effective_video_prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="Scene has no video prompt. Set one first.")
    if not scene.reference_image:
        raise HTTPException(status_code=400, detail="Scene has no reference image. Generate one first.")

    # Resolve audio
    audio_file = proj.config.song.audio_file
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file in project config.")
    audio_path = proj.resolve_path(audio_file)
    segment = proj.resolve_path(scene.audio_segment) if scene.audio_segment else audio_path

    # Determine engine type
    engine_type = VideoEngineType(req.engine) if req.engine else (scene.video_engine or proj.config.video_engine)

    # Pre-compute sub-clip frame plan
    constraints = get_constraints(engine_type.value)
    plan_subclips([scene], constraints, proj.paths.segments_dir, proj.paths.sub_segments_dir)

    if engine_type == VideoEngineType.HUNYUAN_AVATAR:
        config = proj.config.hunyuan_avatar.model_copy()
        config.infer_steps = 10
        config.image_size = 320
        engine = create_video_engine(config, engine_type=engine_type)
    else:
        from musicvision.utils.gpu import detect_devices

        config = HumoConfig.from_quality("preview")
        device_map = detect_devices()
        engine = create_video_engine(config, device_map=device_map, engine_type=engine_type)

    # Resolve pre-computed sub-clip audio paths
    subclip_audio = None
    if scene.generation_audio_segments and len(scene.generation_audio_segments) > 1:
        subclip_audio = [
            proj.resolve_path(p) if not Path(p).is_absolute() else Path(p)
            for p in scene.generation_audio_segments
        ]

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
            scene.video_clip = None
        else:
            scene.video_clip = f"clips/{scene.id}.mp4"
            scene.sub_clips = []

        scene.video_status = ApprovalStatus.PENDING
    finally:
        engine.unload()

    proj.save_scenes()
    return scene


# ---------------------------------------------------------------------------
# Pipeline stage endpoints (stubs — will call into engine modules)
# ---------------------------------------------------------------------------

@app.post("/api/pipeline/intake")
async def run_intake(
    use_llm: bool = True,
    skip_transcription: bool = False,
    use_vocal_separation: bool = False,
):
    """Stage 1: Analyze audio, transcribe, segment into scenes."""
    from musicvision.intake.pipeline import run_intake as _run_intake
    from musicvision.utils.gpu import detect_devices

    proj = get_project()
    device_map = detect_devices()
    scene_list = _run_intake(
        project=proj,
        use_llm_segmentation=use_llm,
        device_map=device_map,
        skip_transcription=skip_transcription,
        use_vocal_separation=use_vocal_separation,
    )
    return {"status": "complete", "scene_count": len(scene_list.scenes)}


@app.post("/api/pipeline/generate-images")
async def generate_images(req: GenerateRequest):
    """Stage 2: Generate reference images for specified scenes (or all)."""
    from musicvision.imaging import create_engine
    from musicvision.imaging.prompt_generator import generate_image_prompt
    from musicvision.utils.gpu import detect_devices

    proj = get_project()

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

    # Create engine and generate
    device_map = detect_devices()
    engine = create_engine(proj.config.image_gen, device_map)
    engine.load()

    generated: list[str] = []
    failed: list[dict] = []

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
                # Find LoRA from first character that has one
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
            except Exception as exc:
                log.error("Image generation failed for %s: %s", scene.id, exc)
                failed.append({"scene_id": scene.id, "error": str(exc)})
    finally:
        engine.unload()

    proj.save_scenes()

    return {
        "status": "complete",
        "generated": generated,
        "failed": failed,
        "total": len(scenes),
    }


@app.post("/api/pipeline/generate-videos")
async def generate_videos(req: GenerateRequest):
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
        etype = scene.video_engine or proj.config.video_engine
        engine_groups[etype].append(scene)

    generated: list[str] = []
    failed: list[dict] = []

    for engine_type, group_scenes in engine_groups.items():
        # Pre-compute sub-clip frame plans for this engine group
        constraints = get_constraints(engine_type.value)
        plan_subclips(group_scenes, constraints, proj.paths.segments_dir, proj.paths.sub_segments_dir)

        # Create engine for this group
        if engine_type == VideoEngineType.HUNYUAN_AVATAR:
            engine = create_video_engine(proj.config.hunyuan_avatar, engine_type=engine_type)
        else:
            from musicvision.utils.gpu import detect_devices
            device_map = detect_devices()
            engine = create_video_engine(proj.config.humo, device_map=device_map, engine_type=engine_type)

        engine.load()

        try:
            for scene in group_scenes:
                try:
                    ref_image = proj.resolve_path(scene.reference_image)
                    segment = proj.resolve_path(scene.audio_segment) if scene.audio_segment else audio_path

                    # Resolve pre-computed sub-clip audio paths
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
                    else:
                        scene.video_clip = f"clips/{scene.id}.mp4"

                    generated.append(scene.id)

                except Exception as exc:
                    log.error("Video generation failed for %s: %s", scene.id, exc)
                    failed.append({"scene_id": scene.id, "error": str(exc)})
        finally:
            engine.unload()

    proj.save_scenes()

    return {
        "status": "complete",
        "generated": generated,
        "failed": failed,
        "total": len(scenes),
    }


class AssembleRequest(BaseModel):
    approved_only: bool = False
    export_edl: bool = True
    export_fcpxml: bool = True


@app.post("/api/pipeline/assemble")
async def assemble(req: AssembleRequest = AssembleRequest()):
    """Stage 4: Concatenate clips, sync audio, export EDL/FCPXML."""
    from musicvision.assembly.concatenator import assemble_rough_cut
    from musicvision.assembly.exporter import export_edl, export_fcpxml

    proj = get_project()
    audio_file = proj.config.song.audio_file
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file set in project config.")

    audio_path = proj.resolve_path(audio_file)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")

    try:
        rough_cut = assemble_rough_cut(
            scenes=proj.scenes,
            paths=proj.paths,
            original_audio=audio_path,
            approved_only=req.approved_only,
        )
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    result: dict = {
        "status": "complete",
        "rough_cut": str(rough_cut.relative_to(proj.paths.root)),
        "clip_count": len([s for s in proj.scenes.scenes if s.video_clip]),
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


# ---------------------------------------------------------------------------
# Static file serving for generated assets
# ---------------------------------------------------------------------------

def mount_project_files(project_dir: Path) -> None:
    """Mount the project directory as static files so the frontend can load images/clips."""
    # Remove existing mount if present (allows re-mounting when opening a different project)
    app.routes[:] = [r for r in app.routes if getattr(r, "name", None) != "project_files"]
    app.mount("/files", StaticFiles(directory=str(project_dir)), name="project_files")
