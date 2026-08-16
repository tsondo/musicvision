# AGENT_INTERFACE_SPEC.md — Job Model + MCP Server

**Status:** Draft for review
**Depends on:** GPU verification of the new two-GPU topology (in progress)
**Supersedes:** the `TODO(job-model)` markers in `api/app.py`

---

## 1. Purpose

Invert the MusicVision interface: today a human drives the pipeline through the
React GUI and the API blocks on GPU work; the target is an **LLM agent driving
the pipeline headlessly through MCP tools**, with the web UI demoted to a
review surface. Both the agent and the UI operate through the same API, so
human approval decisions are visible to the agent as structured evidence.

Three deliverables, in dependency order:

1. **Job model** — long-running operations become jobs: submit → id → poll.
   This is the prerequisite for everything else.
2. **MCP server** — a thin adapter exposing the API as agent tools.
3. **Review write-back** — approval verbs exposed as first-class API/MCP
   operations so decisions flow through one channel.

## 2. Architecture

```
┌──────────────┐   MCP (stdio)   ┌──────────────┐   HTTP    ┌────────────────────┐
│ Agent daemon  │ ──────────────▶ │ musicvision  │ ────────▶ │ musicvision serve  │
│ (Agent SDK)   │                 │ mcp (adapter)│           │  FastAPI + JobMgr  │
└──────────────┘                 └──────────────┘           │  (owns GPU state)  │
                                                            └─────────┬──────────┘
┌──────────────┐            HTTP (same endpoints)                     │
│ React SPA     │ ────────────────────────────────────────────────────┤
│ (review UI)   │                                            ┌────────▼─────────┐
└──────────────┘                                            │ core modules      │
                                                            │ intake/ imaging/  │
                                                            │ video/ upscaling/ │
                                                            │ assembly/         │
                                                            └───────────────────┘
```

Key decisions:

- **One process owns the GPUs.** `musicvision serve` is the only process that
  loads models. The MCP server is a **stateless adapter** that translates tool
  calls into HTTP against the running server. It holds no GPU state, no
  project state, and can restart freely. This also means agent and UI can
  never disagree about state — the serve process is the single source of truth.
- **MCP transport is stdio**, launched by the agent host (Claude Code /
  Agent SDK `mcpServers` config) as `musicvision mcp --server http://<host>:8000`.
  HTTP/SSE transport is a later option if the daemon runs off-box; not v1.
- **Pipeline logic stays in core modules** (existing convention). The job
  manager wraps core calls; endpoints submit jobs; the MCP adapter calls
  endpoints. No layer grows logic.

## 3. Job model

### 3.1 Data model (`models.py`)

```python
class JobKind(str, Enum):
    INTAKE = "intake"
    ANALYZE = "analyze"
    GENERATE_DESCRIPTIONS = "generate_descriptions"
    GENERATE_IMAGES = "generate_images"
    GENERATE_VIDEO_DESCRIPTIONS = "generate_video_descriptions"
    GENERATE_VIDEOS = "generate_videos"
    UPSCALE = "upscale"
    ASSEMBLE = "assemble"
    REGENERATE_IMAGE = "regenerate_image"      # single-scene
    REGENERATE_VIDEO = "regenerate_video"      # single-scene
    UPSCALE_SCENE = "upscale_scene"            # single-scene

class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobProgress(BaseModel):
    current: int = 0            # units completed (scenes, sub-clips, steps)
    total: int = 0              # 0 = indeterminate
    unit: str = "scenes"        # "scenes" | "sub_clips" | "steps"
    message: str = ""           # human-readable, e.g. "scene_004: denoising 12/30"
    scene_id: str | None = None # scene currently being processed

class JobError(BaseModel):
    code: str                   # machine-readable, see §5
    message: str                # human-readable
    detail: dict = {}           # traceback tail, scene id, VRAM snapshot, ...

class Job(BaseModel):
    id: str                     # "job_" + 8 hex chars
    kind: JobKind
    params: dict = {}           # request payload as submitted (scene_ids, model, ...)
    state: JobState = JobState.QUEUED
    progress: JobProgress = JobProgress()
    error: JobError | None = None
    result: dict = {}           # kind-specific summary on success
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
```

### 3.2 Execution semantics

- **Single worker, serial execution.** One background thread consumes a FIFO
  queue. This encodes the existing hard constraint — image and video engines
  never run simultaneously (same VRAM) — structurally instead of by
  convention. Concurrency within a job (e.g. batch images) stays inside the
  core module.
- Jobs run via the existing `asyncio.to_thread()` pattern; the event loop
  stays free for polling, scene reads, and approvals during GPU work.
- **Progress callback:** core-module entry points that loop over scenes gain
  an optional `on_progress: Callable[[JobProgress], None] | None = None`
  parameter. Absent → behavior unchanged (CLI unaffected). The job manager
  passes a callback that updates `job.progress`. Do NOT thread callbacks into
  engine internals in v1 — per-scene granularity is enough for the agent;
  per-denoising-step is a later refinement behind the same interface.
- **Cancellation is cooperative and boundary-checked:** a cancel request sets
  a flag the worker checks *between scenes / sub-clips*, never mid-denoise.
  `QUEUED` jobs cancel immediately. A `RUNNING` job that observes the flag
  finishes its current scene, keeps completed artifacts, and exits as
  `CANCELLED` with `result` reporting partial completion.
- **Persistence:** jobs live in memory plus an append-only
  `<project>/jobs.jsonl` journal (one line per state transition). On serve
  restart, any journal entry left `RUNNING`/`QUEUED` is rewritten as `FAILED`
  with code `server_restarted` — jobs do not survive restarts, but their
  history and failure reason do. The journal doubles as the agent's evidence
  stream of what ran and why it stopped.
- **Queue depth guard:** submitting a job whose kind is already
  `QUEUED`/`RUNNING` for the same scope returns `409 job_already_active`
  with the existing job id in `detail` — agents retry by polling, not by
  resubmitting.

### 3.3 New module

`src/musicvision/jobs.py` — `JobManager` (queue, worker thread, journal,
callback plumbing). No torch imports at module level. Registered on the
FastAPI app at startup; CLI does not use it (CLI remains direct + blocking,
same core calls).

## 4. API changes

### 4.1 New endpoints

| Endpoint | Behavior |
|---|---|
| `POST /api/jobs` | Body `{kind, params}` → `202 {job}`. Validates params against kind before queueing. |
| `GET /api/jobs/{id}` | Full `Job` — the poll target. |
| `GET /api/jobs?state=&kind=&limit=` | Newest-first job list (memory + journal tail). |
| `POST /api/jobs/{id}/cancel` | Request cancellation (see §3.2). `409` if already terminal. |

### 4.2 Migration of existing long-running endpoints

The eight `TODO(job-model)` endpoints (`/api/pipeline/*`, the per-scene
`regenerate-*` and `upscale`) become thin wrappers: validate → submit job →
return `202 {job}`. **Breaking change, no legacy blocking mode** — the
frontend's `usePipeline` hook migrates from `await longPost()` to
submit + poll in the same phase (§7 Phase 3). Its current side-channel
progress hack (snapshotting clip paths and diffing scene lists) is deleted
in favor of `job.progress`.

Fast endpoints (config, scenes CRUD, approvals, uploads, markers) are
untouched — they stay synchronous.

### 4.3 Structured errors (repo-wide convention, enforced here)

All new/updated endpoints return errors as:

```json
{"error": {"code": "scene_not_found", "message": "Scene scene_012 not found", "detail": {"scene_id": "scene_012"}}}
```

Initial code registry (extend as needed, keep flat):
`project_not_open`, `project_busy`, `scene_not_found`,
`no_scenes_to_process`, `job_not_found`, `job_already_active`,
`job_not_cancellable`, `invalid_job_params`, `engine_load_failed`,
`generation_failed`, `vram_exhausted`, `llm_unavailable`,
`server_restarted`, `unhandled_exception`.

FastAPI's default `HTTPException` bodies don't match this shape; add one
exception handler that wraps them. Existing endpoints migrate opportunistically
(when touched), new code complies from day one.

## 5. MCP tool surface

Entry point: `musicvision mcp --server <url>` (new `cli.py` subcommand; MCP
Python SDK, stdio transport). Tools are deliberately fewer and coarser than
the REST surface — the agent needs verbs, not CRUD symmetry:

| Tool | Maps to | Notes |
|---|---|---|
| `open_project(path)` | `POST /api/projects/open` | |
| `get_project_status()` | config + scenes + latest jobs | One composite call: the agent's "where am I" primitive. Returns stage, scene counts by approval state, active job if any. |
| `list_scenes(filter?)` | `GET /api/scenes` | Filter by approval state / missing artifacts. |
| `get_scene(scene_id)` | `GET /api/scenes/{id}` | |
| `update_scene(scene_id, patch)` | `PATCH /api/scenes/{id}` | Prompts, treatment, engine overrides. |
| `submit_job(kind, params?)` | `POST /api/jobs` | |
| `get_job(job_id)` | `GET /api/jobs/{id}` | The poll target. |
| `cancel_job(job_id)` | `POST /api/jobs/{id}/cancel` | |
| `approve(scene_id, target, note?)` | scene PATCH (`image_status`/`video_status`) | `target: "image" \| "video"`. |
| `reject(scene_id, target, note)` | scene PATCH + note | Note **required** on reject — it's the evidence the agent learns from. |
| `get_artifact(scene_id, kind)` | `/files/...` | Returns a **file path on the shared host** (agent daemon is on the same box in v1), plus metadata (resolution, frames). Base64 image bytes optional via `inline=true` for vision-model review of keyframes — size-capped. |
| `get_style_sheet()` / `update_style_sheet(patch)` | config endpoints | |

Every tool docstring is written as an agent-facing tool description
(existing CLAUDE.md convention: "each endpoint should read as a tool
description" — the MCP layer is where that pays off).

**Explicitly not tools in v1:** project create/import (human sets up projects),
asset CRUD (Phase 6 of ASSET_LIBRARY_SPEC lands first), assemble-and-export
knobs beyond `submit_job(kind="assemble")`.

## 6. Review decisions as evidence

Already in the data model: `Scene.image_status` / `Scene.video_status` /
`SubClip.status` (`pending | approved | rejected`). Two additions:

1. **Rejection notes.** `Scene.image_review_note: str = ""` and
   `Scene.video_review_note: str = ""` — free text, written by `reject`
   (required) and `approve` (optional). New optional fields with defaults →
   `scenes.json` backward compatibility holds (verify per CLAUDE.md rule).
2. **Decision provenance.** `reviewed_by: str = ""` (`"human"` | `"agent"`)
   set by the approval endpoints from an `X-MusicVision-Actor` header the MCP
   adapter always sends and the SPA never does (defaults to `"human"`).

That's the entire v1 evidence layer. The agent reads
rejected-with-notes scenes as likelihood updates for its style priors
(belief-file maintenance happens agent-side, not in this repo). No event
bus, no webhook — the agent polls `get_project_status()` on its schedule.

## 7. Phasing

Each phase lands independently green (pytest + ruff; frontend build for P3).

| Phase | Scope | Test focus |
|---|---|---|
| **P1** | `jobs.py` + `Job` models + 4 job endpoints + error-shape handler. No existing endpoint migrated yet. | Unit-test JobManager with fake (sleep-based) work: states, progress, cancel-between-units, journal replay, restart-marks-failed, duplicate-submit guard. No GPU. |
| **P2** | Migrate the 8 long-running endpoints to job submission; add `on_progress` params to core entry points. | Existing API tests updated to submit+poll; progress callback unit-tested at scene granularity. |
| **P3** | Frontend: `usePipeline` → submit+poll `job.progress`; delete the clip-path-diffing progress hack; job-status strip in PipelineBar. | `npm run build` + manual smoke. |
| **P4** | MCP server (`musicvision mcp`) + approval evidence fields (notes, actor). | Tool-schema round-trip test against a running serve with mocked engines; `scenes.json` migration test. |
| **P5** | Agent-side integration (daemon config, nightly review pass) — **out of repo scope**, tracked in the personal-agent project. | — |

GPU-touching validation (P2's real engines, P4 end-to-end with a live
generate) rides the standard integration scripts on the workstation —
same split as today: unit tests everywhere, `scripts/test_gpu_pipeline.py`
on the box.

## 8. Non-goals (v1)

- SSE/WebSocket progress push — polling is sufficient for both consumers;
  SSE is a UI nicety to revisit after P3 (STATUS.md tracks it separately).
- Multi-project serve, auth, off-LAN access (CLOUD_DEPLOYMENT_SPEC territory).
- Job persistence across restarts (journal preserves history; work restarts).
- Distributed / parallel jobs (single-GPU-owner rule is load-bearing).
- Agent-side logic of any kind in this repo (router, beliefs, scheduling).

## 9. Open questions (decide before P1 merge)

1. **Job id in scene artifacts?** Stamping `generated_by_job: str` on
   Scene/SubClip would tie artifacts to journal entries (nice provenance,
   another migration field). Lean yes — cheap now, painful to retrofit.
2. **`get_project_status()` shape** — exact composite payload deserves a
   15-minute design pass against real agent prompts once the daemon exists;
   ship a best-guess in P4 and expect one revision.
3. **Poll interval guidance** — document a recommended cadence (e.g. 2s
   during RUNNING, 30s idle) in tool descriptions so the agent doesn't
   hammer the serve process mid-inference.
