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

from dotenv import load_dotenv

load_dotenv()  # load .env from cwd (or any parent) before anything reads env vars

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from musicvision.imaging.performer_check import run_performer_check
from musicvision.jobs import JobContext, JobFailureError, JobManager, JobManagerError
from musicvision.models import (
    ApprovalStatus,
    HumoConfig,
    ImageGenConfig,
    ImageModel,
    Job,
    JobKind,
    JobState,
    ProjectConfig,
    Scene,
    SceneBoundary,
    StyleSheet,
    VideoEngineType,
)
from musicvision.video.render_mode import RenderMode, config_for_render_mode, missing_render_weights

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

job_manager = JobManager()


def get_project() -> ProjectService:
    if _project is None:
        raise HTTPException(status_code=400, detail="No project loaded. Create or open a project first.")
    return _project


# ---------------------------------------------------------------------------
# Structured errors (AGENT_INTERFACE_SPEC.md §4.3)
#
# New (job) endpoints return {"error": {code, message, detail}}. Existing
# endpoints keep FastAPI's {"detail": ...} shape until the frontend migrates
# (P3) — client.ts currently reads body.detail.
# ---------------------------------------------------------------------------

_ERROR_HTTP_STATUS = {
    "project_not_open": 400,
    "project_busy": 409,
    "invalid_job_params": 400,
    "job_not_found": 404,
    "job_already_active": 409,
    "job_not_cancellable": 409,
    "scene_not_found": 404,
}


class StructuredApiError(Exception):
    """Endpoint-level error carrying a machine-readable code (spec §4.3)."""

    def __init__(self, code: str, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}


@app.exception_handler(StructuredApiError)
@app.exception_handler(JobManagerError)
async def _structured_error_handler(request: Request, exc: StructuredApiError | JobManagerError):
    return JSONResponse(
        status_code=_ERROR_HTTP_STATUS.get(exc.code, 400),
        content={"error": {"code": exc.code, "message": exc.message, "detail": exc.detail}},
    )


@app.exception_handler(RequestValidationError)
async def _request_validation_handler(request: Request, exc: RequestValidationError):
    """Body/query validation failures in the structured error shape (agent-ready API).

    /api/jobs uses invalid_job_params (400), matching its own param validation;
    an unknown kind also lists the valid kinds. Other endpoints: invalid_request (422)."""
    errors = jsonable_encoder(exc.errors())
    if request.url.path.rstrip("/") == "/api/jobs":
        detail: dict = {"errors": errors}
        if any(list(e.get("loc", []))[-1:] == ["kind"] for e in errors):
            detail["valid_kinds"] = [k.value for k in JobKind]
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "invalid_job_params", "message": "Invalid job request", "detail": detail}},
        )
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "invalid_request", "message": "Invalid request", "detail": {"errors": errors}}},
    )


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
    lyrics: str | None = None
    image_prompt_user_override: str | None = None
    video_prompt_user_override: str | None = None
    image_status: ApprovalStatus | None = None
    video_status: ApprovalStatus | None = None
    lip_sync: bool | None = None
    treatment: str | None = None
    video_engine: str | None = None
    sigma_shift: float | None = None
    notes: str | None = None
    # LTX-2 generated audio mixing
    audio_mode: str | None = None
    generated_audio_volume: float | None = None
    song_duck_volume: float | None = None
    audio_fade_in: float | None = None
    audio_fade_out: float | None = None
    song_duck_fade_in: float | None = None
    song_duck_fade_out: float | None = None


class GenerateImagesRequest(BaseModel):
    scene_ids: list[str] = []  # empty = all scenes
    model: str | None = None   # override project config model (e.g. "z-image-turbo")


class GenerateVideosRequest(BaseModel):
    scene_ids: list[str] = []
    engine: str | None = None  # override project config engine (e.g. "humo", "ltx_video")
    render_mode: RenderMode = "preview"  # see musicvision.video.render_mode (applies to HuMo and LTX-2)


class RegenerateImageRequest(BaseModel):
    model: str | None = None   # "z-image-turbo" | "z-image" | "flux-dev" | "flux-schnell"
    seed: int = -1             # -1 = random


class RegenerateVideoRequest(BaseModel):
    engine: str | None = None  # "humo" | "ltx_video"
    seed: int = -1
    render_mode: RenderMode = "preview"  # see musicvision.video.render_mode (applies to HuMo and LTX-2)


# ---------------------------------------------------------------------------
# Project endpoints
# ---------------------------------------------------------------------------

def _guard_no_active_job() -> None:
    if job_manager.has_active_job():
        raise StructuredApiError(
            "project_busy",
            "A job is queued or running. Wait for it to finish or cancel it first.",
        )


def load_project_at_startup(project_dir: Path) -> ProjectService:
    """Open *project_dir* as the served project before the server starts.

    Used by ``musicvision serve <dir>``. Mirrors /api/projects/open: binds the
    job journal (jobs.jsonl) and mounts project files, so jobs are journaled
    and stale ones from a previous run are failed on startup."""
    global _project
    project_dir = project_dir.resolve()
    _project = ProjectService.open(project_dir)
    job_manager.attach_project(project_dir)
    mount_project_files(project_dir)
    return _project


@app.post("/api/projects/create")
async def create_project(req: CreateProjectRequest):
    global _project
    _guard_no_active_job()
    project_dir = Path(req.directory).resolve()
    job_manager.detach_project()
    _project = ProjectService.create(project_dir, name=req.name)
    job_manager.attach_project(project_dir)
    mount_project_files(project_dir)
    return {"status": "created", "name": req.name, "directory": req.directory}


@app.post("/api/projects/open")
async def open_project(req: OpenProjectRequest):
    global _project
    _guard_no_active_job()
    try:
        project_dir = Path(req.directory).resolve()
        new_project = ProjectService.open(project_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    job_manager.detach_project()
    _project = new_project
    job_manager.attach_project(project_dir)
    mount_project_files(project_dir)
    return {"status": "opened", "name": _project.config.name, "directory": req.directory}


@app.post("/api/projects/close")
async def close_project():
    global _project
    _guard_no_active_job()
    job_manager.detach_project()
    _project = None
    return {"status": "closed"}


# ---------------------------------------------------------------------------
# Jobs (AGENT_INTERFACE_SPEC.md §4.1)
#
# Long-running pipeline operations run as jobs: submit -> id -> poll.
# The /api/pipeline/* and per-scene regenerate/upscale endpoints are thin
# wrappers over the same submit path; handlers are registered at the bottom
# of this module (register_job_handlers).
# ---------------------------------------------------------------------------

class SubmitJobRequest(BaseModel):
    kind: JobKind
    params: dict = {}


def _submit_job(kind: JobKind, params: BaseModel) -> Job:
    """Validate-and-queue shared by every job-shaped endpoint.

    Only checks that cannot be invalidated by an earlier queued job run here
    (project open, scene ids exist). Prerequisites such as reference images
    are checked by the handler when the job runs, so an agent can queue
    generate_images and generate_videos back to back."""
    if _project is None:
        raise StructuredApiError("project_not_open", "No project loaded. Create or open a project first.")
    data = params.model_dump(mode="json")
    if data.get("scene_id"):
        data["scene_ids"] = [data["scene_id"]]  # scope for the duplicate-job guard
    missing = [sid for sid in data.get("scene_ids") or [] if _project.scenes.get_scene(sid) is None]
    if missing:
        raise StructuredApiError(
            "scene_not_found", f"Scene(s) not found: {', '.join(missing)}", {"scene_ids": missing},
        )
    return job_manager.submit(kind, data)


def _raise_if_render_weights_missing(config, render_mode: str) -> None:
    """Fail the job (typed) when the render mode needs weights that are not on disk."""
    missing = missing_render_weights(config)
    if missing:
        raise JobFailureError(
            "missing_prerequisite",
            f"render_mode={render_mode!r} needs LoRA {missing['lora']!r}, which is not downloaded.",
            missing,
        )


def _job_project() -> ProjectService:
    """get_project() for handlers: fails the job with a typed error instead of HTTP 400."""
    if _project is None:
        raise JobFailureError("project_not_open", "No project loaded.")
    return _project


def _job_scenes(proj: ProjectService, scene_ids: list[str]) -> list[Scene]:
    """Resolve scene ids inside a handler (empty = all scenes, in order)."""
    if not scene_ids:
        return list(proj.scenes.scenes)
    missing = [sid for sid in scene_ids if proj.scenes.get_scene(sid) is None]
    if missing:
        raise JobFailureError("scene_not_found", f"Scene(s) not found: {', '.join(missing)}", {"scene_ids": missing})
    return [proj.scenes.get_scene(sid) for sid in scene_ids]


@app.post("/api/jobs", status_code=202)
async def submit_job(req: SubmitJobRequest) -> Job:
    """Submit a long-running pipeline job (generate images/videos, upscale,
    assemble, ...). Returns 202 with the queued Job; poll GET /api/jobs/{id}
    for progress. ``params`` are validated against the kind's schema (the
    same body the matching /api/pipeline/* or per-scene endpoint takes; the
    per-scene kinds also need ``scene_id``). Rejects with job_already_active
    if a job of the same kind is already queued/running for an overlapping
    scene scope."""
    if _project is None:
        raise StructuredApiError("project_not_open", "No project loaded. Create or open a project first.")
    model = _JOB_PARAM_MODELS[req.kind]
    try:
        params = model.model_validate(req.params)
    except ValidationError as e:
        import json as _json

        raise StructuredApiError(
            "invalid_job_params",
            f"Invalid params for job kind {req.kind.value!r}",
            {"kind": req.kind.value, "errors": _json.loads(e.json(include_url=False))},
        )
    return _submit_job(req.kind, params)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> Job:
    """Get a job's current state, progress, result, and error. This is the
    poll target while a job runs. Recommended cadence: ~2s while queued or
    running, stop on a terminal state (succeeded/failed/cancelled)."""
    return job_manager.get(job_id)


@app.get("/api/jobs")
async def list_jobs(state: str | None = None, kind: str | None = None, limit: int = 50) -> list[Job]:
    """List jobs newest-first, including journaled history from previous
    server runs. Optional filters: state (queued/running/succeeded/failed/
    cancelled) and kind (e.g. generate_images)."""
    try:
        state_f = JobState(state) if state else None
        kind_f = JobKind(kind) if kind else None
    except ValueError as e:
        raise StructuredApiError("invalid_job_params", str(e), {"state": state, "kind": kind})
    return job_manager.list(state=state_f, kind=kind_f, limit=limit)


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> Job:
    """Request job cancellation. Queued jobs cancel immediately. A running job
    gets cancel_requested=true at once, finishes its current scene/sub-clip
    (completed artifacts are kept), then ends as cancelled with partial results
    (generated/failed lists where the job kind has them). If that was its last
    unit, it ends as succeeded with cancel_requested=true. 409 if already
    finished."""
    return job_manager.cancel(job_id)


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


@app.put("/api/projects/config/video-type")
async def update_video_type(req: dict):
    from musicvision.models import VideoType
    proj = get_project()
    proj.config.video_type = VideoType(req["video_type"])
    proj.save_config()
    return {"status": "updated", "video_type": proj.config.video_type.value}


@app.get("/api/segment-markers")
async def get_segment_markers():
    """Load persisted segment markers, or derive from existing scenes."""
    import json as _json
    proj = get_project()
    path = proj.paths.input_dir / "segment_markers.json"
    if path.exists():
        data = _json.loads(path.read_text())
        if data.get("markers"):  # skip empty marker files (race condition artifact)
            return data

    # No marker file (or empty) — derive from existing scenes if available
    scenes = proj.scenes.scenes
    log.info("segment-markers: no file, deriving from %d scenes (path=%s)", len(scenes), proj.paths.scenes_file)
    if scenes:
        scenes_sorted = sorted(scenes, key=lambda s: s.time_start)
        # Markers are the boundaries between scenes (exclude song start/end)
        markers = [s.time_end for s in scenes_sorted[:-1]]
        # Persist so it's available next time without re-deriving
        data = {"markers": markers}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(data, indent=2))
        return data

    return {"markers": []}


@app.put("/api/segment-markers")
async def save_segment_markers(body: dict):
    """Persist segment markers to input/segment_markers.json."""
    import json as _json
    proj = get_project()
    path = proj.paths.input_dir / "segment_markers.json"
    path.write_text(_json.dumps(body, indent=2))
    return {"status": "saved"}


@app.get("/api/lyrics-assignments")
async def get_lyrics_assignments():
    """Load persisted lyrics-to-scene assignments, or auto-populate from lyrics file."""
    import json as _json
    proj = get_project()
    path = proj.paths.input_dir / "lyrics_assignments.json"
    if path.exists():
        return _json.loads(path.read_text())

    # Auto-populate: parse lyrics into lines, mark section headers
    import re
    assignments: list[dict] = []
    lyrics_path = (
        proj.resolve_path(proj.config.song.lyrics_file)
        if proj.config.song.lyrics_file
        else None
    )
    if lyrics_path and lyrics_path.exists():
        lyrics_text = lyrics_path.read_text(encoding="utf-8")
        for line in lyrics_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            is_header = bool(re.match(r"^\s*[\[\(].*[\]\)]\s*$", stripped))
            assignments.append({"line": stripped, "scene_indices": [], "is_header": is_header})

    return {"assignments": assignments}


@app.put("/api/lyrics-assignments")
async def save_lyrics_assignments(body: dict):
    """Persist lyrics-to-scene assignments."""
    import json as _json
    proj = get_project()
    path = proj.paths.input_dir / "lyrics_assignments.json"
    path.write_text(_json.dumps(body, indent=2))
    return {"status": "saved"}


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

    if req.lyrics is not None:
        scene.lyrics = req.lyrics
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
    if req.treatment is not None:
        from musicvision.models import SceneTreatment
        scene.treatment = SceneTreatment(req.treatment) if req.treatment != "" else None
    if req.video_engine is not None:
        from musicvision.models import VideoEngineType
        scene.video_engine = VideoEngineType(req.video_engine)
    if req.sigma_shift is not None:
        scene.sigma_shift = max(0.5, min(8.0, req.sigma_shift))
    if req.notes is not None:
        scene.notes = req.notes
    # LTX-2 generated audio mixing fields
    if req.audio_mode is not None:
        from musicvision.models import SceneAudioMode
        scene.audio_mode = SceneAudioMode(req.audio_mode)
    if req.generated_audio_volume is not None:
        scene.generated_audio_volume = max(0.0, min(1.0, req.generated_audio_volume))
    if req.song_duck_volume is not None:
        scene.song_duck_volume = max(0.0, min(1.0, req.song_duck_volume))
    if req.audio_fade_in is not None:
        scene.audio_fade_in = max(0.0, min(2.0, req.audio_fade_in))
    if req.audio_fade_out is not None:
        scene.audio_fade_out = max(0.0, min(2.0, req.audio_fade_out))
    if req.song_duck_fade_in is not None:
        scene.song_duck_fade_in = max(0.0, min(2.0, req.song_duck_fade_in))
    if req.song_duck_fade_out is not None:
        scene.song_duck_fade_out = max(0.0, min(2.0, req.song_duck_fade_out))

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
# Per-scene describe endpoints (LLM prompt generation, no GPU)
# ---------------------------------------------------------------------------

@app.post("/api/scenes/{scene_id}/describe-image")
async def describe_image(scene_id: str):
    """Generate an image prompt for a single scene via LLM."""
    from musicvision.imaging.prompt_generator import generate_image_prompt

    proj = get_project()
    scene = proj.scenes.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    import asyncio

    def _run():
        scene.image_prompt = generate_image_prompt(scene, proj.config)
        proj.save_scenes()
        return scene

    return await asyncio.to_thread(_run)


@app.post("/api/scenes/{scene_id}/describe-video")
async def describe_video(scene_id: str):
    """Generate a video motion prompt for a single scene via LLM."""
    from musicvision.video.prompt_generator import generate_video_prompt

    proj = get_project()
    scene = proj.scenes.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found")

    import asyncio

    def _run():
        scene.video_prompt = generate_video_prompt(scene, proj.config)
        proj.save_scenes()
        return scene

    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Per-scene regeneration endpoints
# ---------------------------------------------------------------------------

class RegenerateImageJobParams(RegenerateImageRequest):
    """Params for the regenerate_image job kind (single scene)."""
    scene_id: str


class RegenerateVideoJobParams(RegenerateVideoRequest):
    """Params for the regenerate_video job kind (single scene)."""
    scene_id: str


@app.post("/api/scenes/{scene_id}/regenerate-image", status_code=202)
async def regenerate_image(scene_id: str, req: RegenerateImageRequest) -> Job:
    """Generate or regenerate one scene's reference image (FLUX / Z-Image).

    Runs as a job: returns 202 with the queued Job. Poll GET /api/jobs/{id};
    on success ``result.scene`` is the updated Scene (image_status reset to
    pending). Writes a missing image prompt via the LLM first."""
    return _submit_job(JobKind.REGENERATE_IMAGE, RegenerateImageJobParams(scene_id=scene_id, **req.model_dump()))


def _job_regenerate_image(params: dict, ctx: JobContext) -> dict:
    from musicvision.assets.consistency import (
        resolve_scene_conditioning,
        should_enable_ip_adapter,
    )
    from musicvision.imaging import create_engine
    from musicvision.imaging.prompt_generator import generate_image_prompt
    from musicvision.utils.gpu import detect_devices

    req = RegenerateImageJobParams.model_validate(params)
    proj = _job_project()
    scene = _job_scenes(proj, [req.scene_id])[0]
    ctx.report(0, 1, message=f"Preparing {scene.id}", scene_id=scene.id)

    # Auto-generate prompt if missing
    prompt = scene.effective_image_prompt
    if not prompt:
        scene.image_prompt = generate_image_prompt(scene, proj.config)
        prompt = scene.effective_image_prompt
        if not prompt:
            raise JobFailureError(
                "prompt_generation_failed", "Failed to generate image prompt.", {"scene_id": scene.id},
            )

    # Build config — override model if provided
    config = proj.config.image_gen.model_copy()
    if req.model:
        config.model = ImageModel(req.model)

    # Resolve dimensions from style sheet
    ss = proj.config.style_sheet
    res_parts = ss.resolution.split("x") if "x" in ss.resolution else ["1280", "720"]
    width, height = int(res_parts[0]), int(res_parts[1])

    # Resolve conditioning from the asset library (LoRA + IP-Adapter + prompt
    # fragments). Replaces the old manual character-LoRA lookup. `config` is
    # already a model_copy, so the IP-Adapter auto-enable never persists.
    cond = resolve_scene_conditioning(scene, ss, config.model, project_root=proj.paths.root)
    prompt = cond.apply_to_prompt(prompt)
    if should_enable_ip_adapter(ss, config.model):
        config.ip_adapter.enabled = True

    ctx.raise_if_cancelled()
    ctx.report(0, 1, message=f"Generating image for {scene.id}", scene_id=scene.id)
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
            lora_path=cond.lora_path,
            lora_weight=cond.lora_weight,
            output_path=output_path,
            seed=seed,
            ip_adapter_images=cond.ip_adapter_images if cond.has_ip_adapter else None,
            ip_adapter_scales=cond.ip_adapter_scales if cond.has_ip_adapter else None,
            ip_adapter_embeddings=cond.ip_adapter_embeddings if cond.ip_adapter_embeddings else None,
        )
        scene.reference_image = f"images/{scene.id}.png"
        scene.image_status = ApprovalStatus.PENDING
    finally:
        engine.unload()
    ctx.report(1, 1, message=f"Checking performer in {scene.id}", scene_id=scene.id)
    warning = run_performer_check(scene, proj.config, output_path)
    proj.save_scenes()
    ctx.report(1, 1, message=f"Generated image for {scene.id}", scene_id=scene.id)
    return {"scene": scene.model_dump(mode="json"), "warnings": [warning] if warning else []}


@app.post("/api/scenes/{scene_id}/regenerate-video", status_code=202)
async def regenerate_video(scene_id: str, req: RegenerateVideoRequest) -> Job:
    """Generate or regenerate one scene's video clip (HuMo / LTX-Video 2).

    Runs as a job: returns 202 with the queued Job. Poll GET /api/jobs/{id};
    on success ``result.scene`` is the updated Scene (video_status reset to
    pending). The scene needs a reference image; that is checked when the
    job runs, so it can be queued right behind a regenerate_image job."""
    return _submit_job(JobKind.REGENERATE_VIDEO, RegenerateVideoJobParams(scene_id=scene_id, **req.model_dump()))


def _job_regenerate_video(params: dict, ctx: JobContext) -> dict:
    from musicvision.engine_registry import (
        frames_to_seconds,
        get_constraints,
        plan_subclips,
        sub_clip_suffixes,
    )
    from musicvision.models import SubClip
    from musicvision.video import create_video_engine
    from musicvision.video.prompt_generator import generate_video_prompt

    req = RegenerateVideoJobParams.model_validate(params)
    proj = _job_project()
    scene = _job_scenes(proj, [req.scene_id])[0]
    ctx.report(0, 1, message=f"Preparing {scene.id}", scene_id=scene.id)

    # Auto-generate prompt if missing
    prompt = scene.effective_video_prompt
    if not prompt:
        scene.video_prompt = generate_video_prompt(scene, proj.config)
        prompt = scene.effective_video_prompt
        if not prompt:
            raise JobFailureError(
                "prompt_generation_failed", "Failed to generate video prompt.", {"scene_id": scene.id},
            )
    if not scene.reference_image:
        raise JobFailureError(
            "missing_prerequisite",
            "Scene has no reference image. Generate one first.",
            {"scene_id": scene.id, "missing": "reference_image"},
        )

    # Resolve audio (silence if lip_sync is off)
    audio_file = proj.config.song.audio_file
    if not audio_file:
        raise JobFailureError("missing_prerequisite", "No audio file in project config.", {"missing": "audio_file"})
    audio_path = proj.resolve_path(audio_file)
    segment = _resolve_scene_audio(proj, scene, audio_path)

    # Determine engine type
    engine_type = VideoEngineType(req.engine) if req.engine else (scene.video_engine or proj.config.video_engine)

    # Pre-compute sub-clip frame plan
    constraints = get_constraints(engine_type.value)
    plan_subclips([scene], constraints, proj.paths.segments_dir, proj.paths.sub_segments_dir)

    # Apply render mode
    from musicvision.utils.gpu import detect_devices

    config = config_for_render_mode(engine_type, proj.config, req.render_mode)
    _raise_if_render_weights_missing(config, req.render_mode)
    if engine_type != VideoEngineType.LTX_VIDEO and scene.sigma_shift is not None:
        config.shift = scene.sigma_shift

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

    ctx.raise_if_cancelled()
    ctx.report(0, 1, message=f"Rendering {scene.id} ({engine_type.value})", scene_id=scene.id)
    device_map = detect_devices()
    engine = create_video_engine(config, device_map=device_map, engine_type=engine_type)
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

        scene.upscaled_clip = None  # new video: any old upscale is stale
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

        # Persist LTX-2 generated audio path (single-clip only —
        # scenes with sub-clips don't support generated audio mixing)
        if len(results) == 1 and results[0].generated_audio_path:
            gap = results[0].generated_audio_path
            scene.generated_audio = str(gap.relative_to(proj.paths.root))
            log.info("Generated audio persisted: %s", scene.generated_audio)
        elif len(results) > 1 and any(r.generated_audio_path for r in results):
            log.info(
                "Scene %s split into %d sub-clips — generated audio not supported, using song only",
                scene.id, len(results),
            )
            scene.generated_audio = None
            from musicvision.models import SceneAudioMode
            scene.audio_mode = SceneAudioMode.SONG_ONLY

        scene.video_status = ApprovalStatus.PENDING
        from musicvision.utils.video import update_scene_resolution
        update_scene_resolution(scene, proj.paths.root)
    finally:
        engine.unload()

    proj.save_scenes()
    ctx.report(1, 1, message=f"Rendered {scene.id}", scene_id=scene.id)
    return {"scene": scene.model_dump(mode="json")}


# ---------------------------------------------------------------------------
# Pipeline stage endpoints (stubs — will call into engine modules)
# ---------------------------------------------------------------------------

class AnalyzeParams(BaseModel):
    """Params for the analyze job kind."""
    skip_transcription: bool = False
    use_vocal_separation: bool = True


@app.post("/api/pipeline/analyze", status_code=202)
async def analyze_audio(
    skip_transcription: bool = False,
    use_vocal_separation: bool = True,
) -> Job:
    """Phase 1: Analyze audio (BPM, Whisper, vocal separation). No scene boundaries.

    Runs as a job: returns 202 with the queued Job. Poll GET /api/jobs/{id};
    on success ``result`` is the AnalysisResult (same shape as GET /api/analysis)."""
    return _submit_job(
        JobKind.ANALYZE,
        AnalyzeParams(skip_transcription=skip_transcription, use_vocal_separation=use_vocal_separation),
    )


def _step_reporter(ctx: JobContext, total: int):
    """on_step callback for intake stages: progress = stages finished out of total."""
    done = 0

    def on_step(message: str) -> None:
        nonlocal done
        ctx.report(min(done, total), total, unit="steps", message=message)
        done += 1

    return on_step


def _job_analyze(params: dict, ctx: JobContext) -> dict:
    from musicvision.intake.pipeline import ANALYZE_STEPS
    from musicvision.intake.pipeline import run_analyze as _run_analyze
    from musicvision.utils.gpu import detect_devices

    p = AnalyzeParams.model_validate(params)
    proj = _job_project()
    ctx.report(0, ANALYZE_STEPS, unit="steps", message="Starting audio analysis")
    device_map = detect_devices()
    result = _run_analyze(
        project=proj,
        device_map=device_map,
        skip_transcription=p.skip_transcription,
        use_vocal_separation=p.use_vocal_separation,
        on_step=_step_reporter(ctx, ANALYZE_STEPS),
    )
    ctx.report(ANALYZE_STEPS, ANALYZE_STEPS, unit="steps", message="Analysis complete")
    return result.model_dump(mode="json")


class LyricsAssignmentItem(BaseModel):
    line: str
    scene_indices: list[int]


class CreateScenesRequest(BaseModel):
    boundaries: list[SceneBoundary]
    snap_to_beats: bool = False
    lyrics_assignments: list[LyricsAssignmentItem] | None = None


@app.post("/api/pipeline/create-scenes")
async def create_scenes(req: CreateScenesRequest):
    """Phase 2: Create scenes from manual waveform editor boundaries."""
    from musicvision.intake.pipeline import create_scenes_from_boundaries

    proj = get_project()
    assignments = (
        [a.model_dump() for a in req.lyrics_assignments]
        if req.lyrics_assignments
        else None
    )

    def _run() -> dict:
        scene_list, lyrics_source = create_scenes_from_boundaries(
            project=proj,
            boundaries=req.boundaries,
            snap_to_beats=req.snap_to_beats,
            lyrics_assignments=assignments,
        )
        return {
            "status": "complete",
            "scene_count": len(scene_list.scenes),
            "lyrics_source": lyrics_source,
        }

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


class IntakeParams(BaseModel):
    """Params for the intake job kind."""
    use_llm: bool = True
    skip_transcription: bool = False
    use_vocal_separation: bool = True


@app.post("/api/pipeline/intake", status_code=202)
async def run_intake(
    use_llm: bool = True,
    skip_transcription: bool = False,
    use_vocal_separation: bool = True,
) -> Job:
    """Stage 1: Full intake (analyze + segment in one call). CLI backward-compat.

    Runs as a job: returns 202 with the queued Job; ``result.scene_count`` on success."""
    return _submit_job(
        JobKind.INTAKE,
        IntakeParams(
            use_llm=use_llm,
            skip_transcription=skip_transcription,
            use_vocal_separation=use_vocal_separation,
        ),
    )


def _job_intake(params: dict, ctx: JobContext) -> dict:
    from musicvision.intake.pipeline import ANALYZE_STEPS, SEGMENT_STEPS
    from musicvision.intake.pipeline import run_intake as _run_intake
    from musicvision.utils.gpu import detect_devices

    p = IntakeParams.model_validate(params)
    proj = _job_project()
    total = ANALYZE_STEPS + SEGMENT_STEPS
    ctx.report(0, total, unit="steps", message="Starting intake")
    device_map = detect_devices()
    scene_list = _run_intake(
        project=proj,
        use_llm_segmentation=p.use_llm,
        device_map=device_map,
        skip_transcription=p.skip_transcription,
        use_vocal_separation=p.use_vocal_separation,
        on_step=_step_reporter(ctx, total),
    )
    ctx.report(total, total, unit="steps", message=f"Intake complete: {len(scene_list.scenes)} scenes")
    return {"status": "complete", "scene_count": len(scene_list.scenes)}


class GenerateDescriptionsRequest(BaseModel):
    scene_ids: list[str] = []


def _run_prompt_batch(
    need_prompt: list[Scene],
    ctx: JobContext,
    label: str,
    batch_fn,
    single_fn,
    assign,
) -> dict:
    """Shared LLM prompt loop for image and motion descriptions.

    Tries one batch call first (coherent prompts across scenes), falls back
    to one call per scene. Progress is per scene; cancel is checked between
    scenes in the fallback path."""
    proj = _job_project()
    total = len(need_prompt)
    generated: list[str] = []
    failed: list[dict] = []
    ctx.partial_result(generated=generated, failed=failed)
    if not need_prompt:
        return {"status": "complete", "generated": [], "failed": [], "total": 0}

    ctx.report(0, total, message=f"Writing {total} {label} prompt(s)")
    try:
        prompts = batch_fn(need_prompt, proj.config)
        for scene, prompt in zip(need_prompt, prompts):
            assign(scene, prompt)
            generated.append(scene.id)
        proj.save_scenes()
        ctx.report(total, total, message=f"Wrote {total} {label} prompt(s)")
    except Exception as batch_exc:
        log.warning("Batch %s prompt generation failed: %s — falling back to individual", label, batch_exc)
        for i, scene in enumerate(need_prompt):
            ctx.raise_if_cancelled()
            ctx.report(i, total, message=f"Writing {label} prompt for {scene.id}", scene_id=scene.id)
            try:
                assign(scene, single_fn(scene, proj.config))
                generated.append(scene.id)
                proj.save_scenes()
            except Exception as exc:
                log.error("%s prompt generation failed for %s: %s", label, scene.id, exc)
                failed.append({"scene_id": scene.id, "error": str(exc)})
            ctx.report(i + 1, total, message=f"Wrote {label} prompt for {scene.id}", scene_id=scene.id)

    return {"status": "complete", "generated": generated, "failed": failed, "total": total}


@app.post("/api/pipeline/generate-descriptions", status_code=202)
async def generate_descriptions(req: GenerateDescriptionsRequest) -> Job:
    """Write image prompts via the LLM for scenes that don't have one (no GPU).

    Runs as a job: returns 202 with the queued Job. ``result`` on success:
    {generated: [scene ids], failed: [{scene_id, error}], total}."""
    return _submit_job(JobKind.GENERATE_DESCRIPTIONS, req)


def _job_generate_descriptions(params: dict, ctx: JobContext) -> dict:
    from musicvision.imaging.prompt_generator import (
        generate_image_prompt,
        generate_image_prompts_batch,
    )

    req = GenerateDescriptionsRequest.model_validate(params)
    proj = _job_project()
    scenes = _job_scenes(proj, req.scene_ids)
    need_prompt = [s for s in scenes if not s.effective_image_prompt]

    def assign(scene: Scene, prompt: str) -> None:
        scene.image_prompt = prompt

    return _run_prompt_batch(
        need_prompt, ctx, "image", generate_image_prompts_batch, generate_image_prompt, assign,
    )


@app.post("/api/pipeline/generate-images", status_code=202)
async def generate_images(req: GenerateImagesRequest) -> Job:
    """Generate reference images for scenes (all scenes when scene_ids is empty).

    Runs as a job: returns 202 with the queued Job. Progress is per scene
    (``progress.current/total``, ``progress.scene_id``); cancel takes effect
    between scenes. ``result`` on success: {generated, failed, total}.
    ``model`` overrides and persists the project's image model."""
    return _submit_job(JobKind.GENERATE_IMAGES, req)


def _job_generate_images(params: dict, ctx: JobContext) -> dict:
    from musicvision.assets.consistency import (
        conditioning_sort_key,
        resolve_scene_conditioning,
        should_enable_ip_adapter,
    )
    from musicvision.imaging import create_engine
    from musicvision.imaging.prompt_generator import generate_image_prompt
    from musicvision.utils.gpu import detect_devices

    req = GenerateImagesRequest.model_validate(params)
    proj = _job_project()

    # Apply model override if provided
    if req.model:
        proj.config.image_gen.model = ImageModel(req.model)
        proj.save_config()

    scenes = _job_scenes(proj, req.scene_ids)
    if not scenes:
        raise JobFailureError("no_scenes_to_process", "No scenes to process")
    total = len(scenes)

    # Auto-generate prompts for any scenes still missing one (backward compat with CLI/per-scene regen)
    for scene in scenes:
        if not scene.effective_image_prompt:
            ctx.raise_if_cancelled()
            ctx.report(0, total, message=f"Writing image prompt for {scene.id}", scene_id=scene.id)
            scene.image_prompt = generate_image_prompt(scene, proj.config)

    # Resolve style sheet dimensions
    ss = proj.config.style_sheet
    res_parts = ss.resolution.split("x") if "x" in ss.resolution else ["1280", "720"]
    width, height = int(res_parts[0]), int(res_parts[1])

    # Resolve per-scene conditioning from the asset library (LoRA + IP-Adapter
    # + prompt fragments). Replaces the old manual character-LoRA lookup.
    conds = {
        s.id: resolve_scene_conditioning(
            s, ss, proj.config.image_gen.model, project_root=proj.paths.root,
        )
        for s in scenes
    }

    # Engine config copy: auto-enable IP-Adapter before load() (must precede
    # cpu-offload — see FluxEngine load-order audit) without persisting the
    # runtime flag into project.yaml.
    engine_config = proj.config.image_gen.model_copy(deep=True)
    if should_enable_ip_adapter(ss, engine_config.model):
        engine_config.ip_adapter.enabled = True

    ctx.raise_if_cancelled()
    ctx.report(0, total, message=f"Loading {engine_config.model.value}")
    device_map = detect_devices()
    engine = create_engine(engine_config, device_map)
    engine.load()

    generated: list[str] = []
    failed: list[dict] = []
    warnings: list[dict] = []
    ctx.partial_result(generated=generated, failed=failed, warnings=warnings)

    try:
        # Group by LoRA (minimize fuse/unfuse swaps), then by IP-Adapter
        # reference set (predictable ordering; IPA needs no load cycles).
        sorted_scenes = sorted(scenes, key=lambda s: conditioning_sort_key(conds[s.id]))

        for i, scene in enumerate(sorted_scenes):
            ctx.raise_if_cancelled()  # between scenes only — never mid-denoise
            ctx.report(i, total, message=f"Generating image for {scene.id}", scene_id=scene.id)
            try:
                cond = conds[scene.id]
                output_path = proj.paths.image_path(scene.id)
                engine.generate(
                    prompt=cond.apply_to_prompt(scene.effective_image_prompt),
                    width=width,
                    height=height,
                    lora_path=cond.lora_path,
                    lora_weight=cond.lora_weight,
                    output_path=output_path,
                    ip_adapter_images=cond.ip_adapter_images if cond.has_ip_adapter else None,
                    ip_adapter_scales=cond.ip_adapter_scales if cond.has_ip_adapter else None,
                    ip_adapter_embeddings=cond.ip_adapter_embeddings if cond.ip_adapter_embeddings else None,
                )
                scene.reference_image = f"images/{scene.id}.png"
                generated.append(scene.id)
                if warning := run_performer_check(scene, proj.config, output_path):
                    warnings.append(warning)
                proj.save_scenes()
            except Exception as exc:
                log.error("Image generation failed for %s: %s", scene.id, exc)
                failed.append({"scene_id": scene.id, "error": str(exc)})
            ctx.report(i + 1, total, message=f"Finished {scene.id}", scene_id=scene.id)
    finally:
        engine.unload()

    return {
        "status": "complete",
        "generated": generated,
        "failed": failed,
        "warnings": warnings,
        "total": total,
    }


class GenerateVideoDescriptionsRequest(BaseModel):
    scene_ids: list[str] = []


@app.post("/api/pipeline/generate-video-descriptions", status_code=202)
async def generate_video_descriptions(req: GenerateVideoDescriptionsRequest) -> Job:
    """Write video motion prompts via the LLM for scenes that don't have one (no GPU).

    Runs as a job: returns 202 with the queued Job. ``result`` on success:
    {generated, failed, total}."""
    return _submit_job(JobKind.GENERATE_VIDEO_DESCRIPTIONS, req)


def _job_generate_video_descriptions(params: dict, ctx: JobContext) -> dict:
    from musicvision.video.prompt_generator import (
        generate_video_prompt,
        generate_video_prompts_batch,
    )

    req = GenerateVideoDescriptionsRequest.model_validate(params)
    proj = _job_project()
    scenes = _job_scenes(proj, req.scene_ids)
    need_prompt = [s for s in scenes if not s.effective_video_prompt]

    def assign(scene: Scene, prompt: str) -> None:
        scene.video_prompt = prompt

    return _run_prompt_batch(
        need_prompt, ctx, "motion", generate_video_prompts_batch, generate_video_prompt, assign,
    )


@app.post("/api/pipeline/generate-videos", status_code=202)
async def generate_videos(req: GenerateVideosRequest) -> Job:
    """Stage 3: Generate video clips for specified scenes (or all).

    Runs as a job: returns 202 with the queued Job. Scenes are grouped by
    their ``video_engine`` (falling back to the project default, or
    ``engine`` for this run) and each group runs with one engine loaded.
    Progress is per scene; cancel takes effect between scenes. Reference
    images are checked when the job runs, so this can be queued right
    behind a generate_images job. ``result`` on success: {generated,
    failed, total, vram_warnings?}."""
    return _submit_job(JobKind.GENERATE_VIDEOS, req)


def _job_generate_videos(params: dict, ctx: JobContext) -> dict:
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

    req = GenerateVideosRequest.model_validate(params)
    proj = _job_project()

    # Engine override for this run only (don't persist to project.yaml)
    run_engine = VideoEngineType(req.engine) if req.engine else proj.config.video_engine

    scenes = _job_scenes(proj, req.scene_ids)
    if not scenes:
        raise JobFailureError("no_scenes_to_process", "No scenes to process")
    total = len(scenes)

    # Check that scenes have reference images (Stage 2 must run first)
    missing_images = [s.id for s in scenes if not s.reference_image]
    if missing_images:
        raise JobFailureError(
            "missing_prerequisite",
            f"Scenes missing reference images (run generate-images first): {missing_images}",
            {"missing": "reference_image", "scene_ids": missing_images},
        )

    # Generate video prompts for scenes that don't have one yet
    for scene in scenes:
        if not scene.effective_video_prompt:
            ctx.raise_if_cancelled()
            ctx.report(0, total, message=f"Writing motion prompt for {scene.id}", scene_id=scene.id)
            scene.video_prompt = generate_video_prompt(scene, proj.config)

    # Resolve audio file
    audio_file = proj.config.song.audio_file
    if not audio_file:
        raise JobFailureError("missing_prerequisite", "No audio file set in project config.", {"missing": "audio_file"})
    audio_path = proj.resolve_path(audio_file)
    if not audio_path.exists():
        raise JobFailureError(
            "missing_prerequisite", f"Audio file not found: {audio_path}", {"missing": "audio_file"},
        )

    # Group scenes by engine type
    engine_groups: dict[VideoEngineType, list] = defaultdict(list)
    for scene in sorted(scenes, key=lambda s: s.order):
        etype = scene.video_engine or run_engine
        engine_groups[etype].append(scene)

    from musicvision.utils.gpu import _oom_suggestion, detect_devices, estimate_vram_gb, is_oom_error

    vram_warnings: list[dict] = []

    # --- Pre-flight VRAM check (advisory, does not block) ---
    for engine_type, group_scenes in engine_groups.items():
        engine_config = proj.config.ltx_video if engine_type == VideoEngineType.LTX_VIDEO else proj.config.humo
        estimated = estimate_vram_gb(
            engine_type.value,
            image_size=getattr(engine_config, "image_size", 0),
            cpu_offload=getattr(engine_config, "cpu_offload", True),
        )
        if estimated > 0:
            try:
                import torch
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                available_gb = round(free_bytes / 1024**3, 1)
            except Exception:
                available_gb = None

            if available_gb is not None and estimated + 2.0 > available_gb:
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

    # Render mode is applied to per-engine copies; proj.config is never mutated.
    render_mode = req.render_mode
    run_configs = {et: config_for_render_mode(et, proj.config, render_mode) for et in engine_groups}
    for cfg in run_configs.values():
        _raise_if_render_weights_missing(cfg, render_mode)
    log.info("Render mode: %s", render_mode)

    generated: list[str] = []
    failed: list[dict] = []
    ctx.partial_result(generated=generated, failed=failed)
    done = 0

    for engine_type, group_scenes in engine_groups.items():
        ctx.raise_if_cancelled()
        constraints = get_constraints(engine_type.value)
        plan_subclips(group_scenes, constraints, proj.paths.segments_dir, proj.paths.sub_segments_dir)

        ctx.report(done, total, message=f"Loading {engine_type.value}")
        device_map = detect_devices()
        run_config = run_configs[engine_type]
        default_shift = getattr(run_config, "shift", None)
        engine = create_video_engine(run_config, device_map=device_map, engine_type=engine_type)

        engine.load()
        consecutive_ooms = 0

        try:
            for scene in group_scenes:
                ctx.raise_if_cancelled()  # between scenes only — never mid-denoise
                ctx.report(done, total, message=f"Rendering {scene.id} ({engine_type.value})", scene_id=scene.id)
                if consecutive_ooms >= 2:
                    engine_config = run_config
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
                    done += 1
                    ctx.report(done, total, message=f"Skipped {scene.id}", scene_id=scene.id)
                    continue

                try:
                    import random
                    if scene.video_status.value == "approved" and scene.video_seed is not None:
                        scene_seed = scene.video_seed
                    else:
                        scene_seed = random.randint(0, 2**31 - 1)
                        scene.video_seed = scene_seed

                    # The engine holds run_config, so per-scene values go on it.
                    run_config.seed = scene_seed
                    if engine_type != VideoEngineType.LTX_VIDEO:
                        run_config.shift = scene.sigma_shift if scene.sigma_shift is not None else default_shift

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

                    scene.upscaled_clip = None  # new video: any old upscale is stale
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
                        # Multi-sub-clip scenes cannot use generated audio
                        if any(r.generated_audio_path for r in results):
                            from musicvision.models import SceneAudioMode
                            log.info(
                                "Scene %s split into %d sub-clips — forcing song_only audio mode",
                                scene.id, len(results),
                            )
                            scene.generated_audio = None
                            scene.audio_mode = SceneAudioMode.SONG_ONLY
                    else:
                        scene.video_clip = f"clips/{scene.id}.mp4"
                        # Persist generated audio for single-clip LTX-2 scenes
                        if results[0].generated_audio_path:
                            gap = results[0].generated_audio_path
                            scene.generated_audio = str(gap.relative_to(proj.paths.root))
                            log.info("Generated audio persisted: %s", scene.generated_audio)

                    from musicvision.utils.video import update_scene_resolution
                    update_scene_resolution(scene, proj.paths.root)
                    generated.append(scene.id)
                    consecutive_ooms = 0
                    proj.save_scenes()

                except Exception as exc:
                    if is_oom_error(exc):
                        consecutive_ooms += 1
                        engine_config = run_config
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
                done += 1
                ctx.report(done, total, message=f"Finished {scene.id}", scene_id=scene.id)
        finally:
            engine.unload()

    result: dict = {
        "status": "complete",
        "generated": generated,
        "failed": failed,
        "total": total,
    }
    if vram_warnings:
        result["vram_warnings"] = vram_warnings
    return result


class UpscaleRequest(BaseModel):
    scene_ids: list[str] = []
    resolution: str | None = None       # "720p" | "1080p" | "1440p" | "4k"
    upscaler: str | None = None         # override: "ltx_spatial" | "seedvr2" | "real_esrgan"
    render_mode: str = "final"          # "preview" | "final"
    force: bool = False                 # redo scenes already upscaled at the target resolution


class UpscaleSceneJobParams(UpscaleRequest):
    """Params for the upscale_scene job kind (single scene)."""
    scene_id: str


class AssembleRequest(BaseModel):
    approved_only: bool = False
    export_edl: bool = True
    export_fcpxml: bool = True


def _run_upscale(
    req: UpscaleRequest, scene_ids: list[str] | None, ctx: JobContext, *, skip_current: bool,
) -> dict:
    """Shared body for the upscale and upscale_scene jobs."""
    from musicvision.jobs import JobCancelledError
    from musicvision.models import TargetResolution, UpscalerType
    from musicvision.upscaling.pipeline import upscale_clips

    proj = _job_project()

    # Overrides apply to a copy; proj.config is never mutated by a job.
    upscaler_config = proj.config.upscaler.model_copy(deep=True)
    if req.resolution:
        upscaler_config.target_resolution = TargetResolution(req.resolution)
    if req.upscaler:
        upscaler_config.upscaler_override = UpscalerType(req.upscaler)

    device_map = None
    try:
        from musicvision.utils.gpu import detect_devices
        device_map = detect_devices()
    except Exception:
        pass

    ctx.report(0, 0, message="Preparing upscale")
    result = upscale_clips(
        scenes=proj.scenes,
        paths=proj.paths,
        upscaler_config=upscaler_config,
        default_engine=proj.config.video_engine,
        render_mode=req.render_mode,
        scene_ids=scene_ids,
        device_map=device_map,
        on_progress=lambda p: ctx.report(p.current, p.total, p.unit, p.message, p.scene_id),
        should_cancel=ctx.cancelled,
        skip_current=skip_current,
    )
    proj.save_scenes()
    if result.get("cancelled"):
        ctx.partial_result(upscaled=result["upscaled"], failed=result["failed"], skipped=result["skipped"])
        raise JobCancelledError()
    return {
        "status": "complete",
        "upscaled": result["upscaled"],
        "failed": result["failed"],
        "skipped": result["skipped"],
        "total": len(result["upscaled"]) + len(result["failed"]),
    }


@app.post("/api/pipeline/upscale", status_code=202)
async def upscale_videos(req: UpscaleRequest) -> Job:
    """Stage 4b: Upscale video clips to target resolution (all scenes with
    clips when scene_ids is empty).

    Runs as a job: returns 202 with the queued Job. Progress is per scene;
    cancel takes effect between scenes. ``result`` on success:
    {upscaled, failed, total}."""
    return _submit_job(JobKind.UPSCALE, req)


def _job_upscale(params: dict, ctx: JobContext) -> dict:
    req = UpscaleRequest.model_validate(params)
    return _run_upscale(req, req.scene_ids or None, ctx, skip_current=not req.force)


@app.post("/api/scenes/{scene_id}/upscale", status_code=202)
async def upscale_scene(scene_id: str, req: UpscaleRequest) -> Job:
    """Upscale one scene's video clip.

    Runs as a job: returns 202 with the queued Job; ``result`` on success:
    {upscaled, failed, total}. The scene's clip is checked when the job runs."""
    params = UpscaleSceneJobParams(scene_id=scene_id, **req.model_dump(exclude={"scene_ids"}))
    return _submit_job(JobKind.UPSCALE_SCENE, params)


def _job_upscale_scene(params: dict, ctx: JobContext) -> dict:
    req = UpscaleSceneJobParams.model_validate(params)
    proj = _job_project()
    scene = _job_scenes(proj, [req.scene_id])[0]
    if not scene.video_clip and not scene.sub_clips:
        raise JobFailureError(
            "missing_prerequisite",
            "Scene has no video clip to upscale",
            {"scene_id": scene.id, "missing": "video_clip"},
        )
    return _run_upscale(req, [scene.id], ctx, skip_current=False)  # explicit per-scene request: always redo


@app.post("/api/pipeline/assemble", status_code=202)
async def assemble(req: AssembleRequest = AssembleRequest()) -> Job:
    """Stage 5: Concatenate clips, sync audio, export EDL/FCPXML.

    Runs as a job: returns 202 with the queued Job. ``result`` on success:
    {rough_cut, clip_count, total_scenes, duration_seconds,
    video_duration_seconds, drift_seconds, warnings, output_dir, edl?, fcpxml?}.
    ``warnings`` is a list of {code, message, detail}; ``av_drift`` means the
    rough cut does not match the song length (e.g. gaps between scenes), so
    lip sync drifts and the end is truncated. The job still succeeds."""
    return _submit_job(JobKind.ASSEMBLE, req)


def _job_assemble(params: dict, ctx: JobContext) -> dict:
    from musicvision.assembly.concatenator import assemble_rough_cut, sync_warnings
    from musicvision.assembly.exporter import export_edl, export_fcpxml
    from musicvision.engine_registry import get_constraints
    from musicvision.utils.audio import get_duration

    req = AssembleRequest.model_validate(params)
    proj = _job_project()
    audio_file = proj.config.song.audio_file
    if not audio_file:
        raise JobFailureError("missing_prerequisite", "No audio file set in project config.", {"missing": "audio_file"})

    audio_path = proj.resolve_path(audio_file)
    if not audio_path.exists():
        raise JobFailureError(
            "missing_prerequisite", f"Audio file not found: {audio_path}", {"missing": "audio_file"},
        )

    ctx.report(0, 0, unit="steps", message="Assembling rough cut")
    try:
        rough_cut = assemble_rough_cut(
            scenes=proj.scenes,
            paths=proj.paths,
            original_audio=audio_path,
            approved_only=req.approved_only,
            fps=get_constraints(proj.config.video_engine.value).fps,
        )
    except (RuntimeError, ValueError) as e:
        raise JobFailureError("assembly_failed", str(e))

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
        "warnings": sync_warnings(proj.scenes, video_dur, get_constraints(proj.config.video_engine.value).fps),
        "output_dir": str(proj.paths.root / "output"),
    }

    if req.export_edl:
        ctx.report(0, 0, unit="steps", message="Exporting EDL")
        edl = export_edl(proj.scenes, proj.paths)
        result["edl"] = str(edl.relative_to(proj.paths.root))

    if req.export_fcpxml:
        ctx.report(0, 0, unit="steps", message="Exporting FCPXML")
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
# Job registration (AGENT_INTERFACE_SPEC.md §4.2)
#
# Every long-running endpoint above is a thin "validate -> submit" wrapper;
# the work runs on the JobManager's single worker thread via these handlers.
# _JOB_PARAM_MODELS validates POST /api/jobs params per kind before queueing.
# ---------------------------------------------------------------------------

_JOB_PARAM_MODELS: dict[JobKind, type[BaseModel]] = {
    JobKind.ANALYZE: AnalyzeParams,
    JobKind.INTAKE: IntakeParams,
    JobKind.GENERATE_DESCRIPTIONS: GenerateDescriptionsRequest,
    JobKind.GENERATE_IMAGES: GenerateImagesRequest,
    JobKind.GENERATE_VIDEO_DESCRIPTIONS: GenerateVideoDescriptionsRequest,
    JobKind.GENERATE_VIDEOS: GenerateVideosRequest,
    JobKind.UPSCALE: UpscaleRequest,
    JobKind.ASSEMBLE: AssembleRequest,
    JobKind.REGENERATE_IMAGE: RegenerateImageJobParams,
    JobKind.REGENERATE_VIDEO: RegenerateVideoJobParams,
    JobKind.UPSCALE_SCENE: UpscaleSceneJobParams,
}

_JOB_HANDLERS = {
    JobKind.ANALYZE: _job_analyze,
    JobKind.INTAKE: _job_intake,
    JobKind.GENERATE_DESCRIPTIONS: _job_generate_descriptions,
    JobKind.GENERATE_IMAGES: _job_generate_images,
    JobKind.GENERATE_VIDEO_DESCRIPTIONS: _job_generate_video_descriptions,
    JobKind.GENERATE_VIDEOS: _job_generate_videos,
    JobKind.UPSCALE: _job_upscale,
    JobKind.ASSEMBLE: _job_assemble,
    JobKind.REGENERATE_IMAGE: _job_regenerate_image,
    JobKind.REGENERATE_VIDEO: _job_regenerate_video,
    JobKind.UPSCALE_SCENE: _job_upscale_scene,
}


def _oom_to_vram_exhausted(handler):
    """Turn a CUDA OOM escaping a handler into a typed vram_exhausted failure."""

    def wrapped(params: dict, ctx: JobContext) -> dict:
        try:
            return handler(params, ctx)
        except JobFailureError:
            raise
        except Exception as exc:
            from musicvision.utils.gpu import is_oom_error

            if is_oom_error(exc):
                raise JobFailureError("vram_exhausted", str(exc), {"exception_type": type(exc).__name__}) from exc
            raise

    return wrapped


def register_job_handlers(manager: JobManager) -> None:
    """Register the real pipeline handler for every JobKind on ``manager``."""
    for kind, handler in _JOB_HANDLERS.items():
        manager.register(kind, _oom_to_vram_exhausted(handler))


register_job_handlers(job_manager)


# ---------------------------------------------------------------------------
# Static file serving for generated assets
# ---------------------------------------------------------------------------

def mount_project_files(project_dir: Path) -> None:
    """Mount the project directory as static files so the frontend can load images/clips."""
    # Remove existing mount if present (allows re-mounting when opening a different project)
    app.routes[:] = [r for r in app.routes if getattr(r, "name", None) != "project_files"]
    app.mount("/files", StaticFiles(directory=str(project_dir)), name="project_files")
