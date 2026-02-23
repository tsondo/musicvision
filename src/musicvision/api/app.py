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
    FluxConfig,
    HumoConfig,
    ProjectConfig,
    Scene,
    StyleSheet,
)
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


# ---------------------------------------------------------------------------
# Project endpoints
# ---------------------------------------------------------------------------

@app.post("/api/projects/create")
async def create_project(req: CreateProjectRequest):
    global _project
    _project = ProjectService.create(Path(req.directory), name=req.name)
    return {"status": "created", "name": req.name, "directory": req.directory}


@app.post("/api/projects/open")
async def open_project(req: OpenProjectRequest):
    global _project
    try:
        _project = ProjectService.open(Path(req.directory))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
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


@app.put("/api/projects/config/flux")
async def update_flux_config(flux: FluxConfig):
    proj = get_project()
    proj.config.flux = flux
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
# Pipeline stage endpoints (stubs — will call into engine modules)
# ---------------------------------------------------------------------------

@app.post("/api/pipeline/intake")
async def run_intake(use_llm: bool = True, skip_transcription: bool = False):
    """Stage 1: Analyze audio, transcribe, segment into scenes."""
    from musicvision.intake.pipeline import run_intake as _run_intake

    proj = get_project()
    scene_list = _run_intake(
        project=proj,
        use_llm_segmentation=use_llm,
        skip_transcription=skip_transcription,
    )
    return {"status": "complete", "scene_count": len(scene_list.scenes)}


@app.post("/api/pipeline/generate-images")
async def generate_images(req: GenerateRequest):
    """Stage 2: Generate reference images for specified scenes (or all)."""
    # TODO: wire to imaging module
    return {"status": "not_implemented", "stage": "generate_images", "scene_ids": req.scene_ids}


@app.post("/api/pipeline/generate-videos")
async def generate_videos(req: GenerateRequest):
    """Stage 3: Generate video clips for specified scenes (or all)."""
    # TODO: wire to video module
    return {"status": "not_implemented", "stage": "generate_videos", "scene_ids": req.scene_ids}


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
    app.mount("/files", StaticFiles(directory=str(project_dir)), name="project_files")
