"""
Job manager for long-running pipeline operations (AGENT_INTERFACE_SPEC.md §3).

Single-worker serial execution: one background thread consumes a FIFO queue.
This encodes the hard constraint that image and video engines never run
simultaneously (they share GPU0 VRAM) structurally instead of by convention.

Handlers are registered per JobKind by the API layer (P2 of the spec); this
module knows nothing about engines or core pipeline modules. A handler is a
callable ``(params: dict, ctx: JobContext) -> dict`` running on the worker
thread. It reports progress via ``ctx.report(...)`` and must call
``ctx.raise_if_cancelled()`` at unit boundaries (between scenes/sub-clips —
never mid-denoise) to support cooperative cancellation.

Jobs do not survive a server restart, but their history does: every state
transition appends the full job snapshot to ``<project>/jobs.jsonl``. On
``attach_project()`` any journaled job left non-terminal is rewritten as
FAILED(server_restarted).
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from musicvision.models import (
    TERMINAL_JOB_STATES,
    Job,
    JobError,
    JobKind,
    JobProgress,
    JobState,
)

log = logging.getLogger(__name__)

JobHandler = Callable[[dict, "JobContext"], dict]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class JobManagerError(Exception):
    """Base for job manager errors. Carries a machine-readable code."""

    def __init__(self, code: str, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}


class UnknownJobKindError(JobManagerError):
    def __init__(self, kind: JobKind) -> None:
        super().__init__(
            "invalid_job_params",
            f"No handler registered for job kind {kind.value!r}",
            {"kind": kind.value},
        )


class DuplicateJobError(JobManagerError):
    def __init__(self, existing: Job) -> None:
        super().__init__(
            "job_already_active",
            f"A {existing.kind.value!r} job is already {existing.state.value} (id={existing.id})",
            {"job_id": existing.id, "state": existing.state.value},
        )


class JobNotFoundError(JobManagerError):
    def __init__(self, job_id: str) -> None:
        super().__init__("job_not_found", f"Job {job_id} not found", {"job_id": job_id})


class JobNotCancellableError(JobManagerError):
    def __init__(self, job: Job) -> None:
        super().__init__(
            "job_not_cancellable",
            f"Job {job.id} is already {job.state.value}",
            {"job_id": job.id, "state": job.state.value},
        )


class JobFailureError(Exception):
    """Raised by handlers to fail a job with a typed error code."""

    def __init__(self, code: str, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}


class JobCancelledError(Exception):
    """Raised inside a handler by ctx.raise_if_cancelled()."""


# ---------------------------------------------------------------------------
# Handler-side context
# ---------------------------------------------------------------------------

class JobContext:
    """Passed to handlers: progress reporting + cooperative cancellation."""

    def __init__(self, manager: JobManager, job_id: str) -> None:
        self._manager = manager
        self._job_id = job_id

    def report(
        self,
        current: int,
        total: int,
        unit: str = "scenes",
        message: str = "",
        scene_id: str | None = None,
    ) -> None:
        self._manager._update_progress(
            self._job_id,
            JobProgress(current=current, total=total, unit=unit, message=message, scene_id=scene_id),
        )

    def cancelled(self) -> bool:
        return self._manager._is_cancel_requested(self._job_id)

    def raise_if_cancelled(self) -> None:
        """Call at unit boundaries (between scenes/sub-clips), never mid-denoise."""
        if self.cancelled():
            raise JobCancelledError()


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

def _scopes_conflict(a: Job, b_kind: JobKind, b_scene_ids: frozenset[str]) -> bool:
    """Same kind conflicts if either targets the whole project or scenes overlap."""
    if a.kind != b_kind:
        return False
    a_scenes = frozenset(a.params.get("scene_ids") or ())
    if not a_scenes or not b_scene_ids:
        return True
    return bool(a_scenes & b_scene_ids)


class JobManager:
    """FIFO queue + single worker thread + per-project jsonl journal."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []            # insertion order, for listing
        self._queue: deque[str] = deque()
        self._handlers: dict[JobKind, JobHandler] = {}
        self._cancel_requested: set[str] = set()
        self._journal_path: Path | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = False
        self._worker: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def register(self, kind: JobKind, handler: JobHandler) -> None:
        self._handlers[kind] = handler

    def attach_project(self, project_root: Path) -> None:
        """Bind the journal to a project and fail any journaled non-terminal jobs."""
        with self._lock:
            self._journal_path = project_root / "jobs.jsonl"
            stale = self._replay_journal_locked()
        for job in stale:
            self._journal_append(job)
            log.warning("Journaled job %s (%s) was non-terminal; marked failed", job.id, job.kind.value)

    def detach_project(self) -> None:
        with self._lock:
            self._journal_path = None
            self._jobs.clear()
            self._order.clear()
            self._queue.clear()
            self._cancel_requested.clear()

    def has_active_job(self) -> bool:
        with self._lock:
            return any(not j.is_terminal for j in self._jobs.values())

    def shutdown(self) -> None:
        """Stop the worker after the current job (tests + serve teardown)."""
        self._stopping = True
        self._wake.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=10)

    # -- public API --------------------------------------------------------

    def submit(self, kind: JobKind, params: dict | None = None) -> Job:
        params = params or {}
        if kind not in self._handlers:
            raise UnknownJobKindError(kind)
        scene_ids = frozenset(params.get("scene_ids") or ())
        with self._lock:
            for job_id in self._order:
                existing = self._jobs[job_id]
                if not existing.is_terminal and _scopes_conflict(existing, kind, scene_ids):
                    raise DuplicateJobError(existing)
            job = Job(
                id=f"job_{uuid.uuid4().hex[:8]}",
                kind=kind,
                params=params,
                created_at=datetime.now(UTC),
            )
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._queue.append(job.id)
            snapshot = job.model_copy(deep=True)
        self._journal_append(snapshot)
        self._ensure_worker()
        self._wake.set()
        return snapshot

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            return job.model_copy(deep=True)

    def list(
        self,
        state: JobState | None = None,
        kind: JobKind | None = None,
        limit: int = 50,
    ) -> list[Job]:
        with self._lock:
            jobs = [self._jobs[jid] for jid in reversed(self._order)]
        if state is not None:
            jobs = [j for j in jobs if j.state == state]
        if kind is not None:
            jobs = [j for j in jobs if j.kind == kind]
        return [j.model_copy(deep=True) for j in jobs[:limit]]

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.is_terminal:
                raise JobNotCancellableError(job)
            if job.state == JobState.QUEUED:
                self._queue.remove(job_id)
                self._finish_locked(job, JobState.CANCELLED)
            else:  # RUNNING — cooperative: worker observes the flag between units
                self._cancel_requested.add(job_id)
            snapshot = job.model_copy(deep=True)
        if snapshot.is_terminal:
            self._journal_append(snapshot)
        return snapshot

    # -- worker ------------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._stopping = False
            self._worker = threading.Thread(target=self._worker_loop, name="musicvision-jobs", daemon=True)
            self._worker.start()

    def _worker_loop(self) -> None:
        while not self._stopping:
            with self._lock:
                job_id = self._queue.popleft() if self._queue else None
            if job_id is None:
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                continue
            self._run_one(job_id)

    def _run_one(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state != JobState.QUEUED:
                return  # cancelled while queued, or project detached
            job.state = JobState.RUNNING
            job.started_at = datetime.now(UTC)
            handler = self._handlers.get(job.kind)
            snapshot = job.model_copy(deep=True)
        self._journal_append(snapshot)

        try:
            if handler is None:  # unregistered between submit and run (shouldn't happen)
                raise JobFailureError("invalid_job_params", f"No handler for {job.kind.value}")
            result = handler(dict(job.params), JobContext(self, job_id))
            outcome, error = JobState.SUCCEEDED, None
        except JobCancelledError:
            result, outcome, error = {}, JobState.CANCELLED, None
        except JobFailureError as e:
            result, outcome = {}, JobState.FAILED
            error = JobError(code=e.code, message=e.message, detail=e.detail)
        except Exception as e:  # noqa: BLE001 — worker must never die
            log.exception("Job %s (%s) crashed", job_id, job.kind.value)
            result, outcome = {}, JobState.FAILED
            error = JobError(
                code="unhandled_exception",
                message=f"{type(e).__name__}: {e}",
                detail={"exception_type": type(e).__name__},
            )

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:  # project detached mid-run
                return
            if outcome == JobState.CANCELLED:
                # Cooperative cancel keeps completed artifacts; handler-reported
                # progress shows how far it got.
                job.result = {"partial": True, "completed_units": job.progress.current}
            else:
                job.result = result if isinstance(result, dict) else {}
            self._finish_locked(job, outcome, error)
            snapshot = job.model_copy(deep=True)
        self._journal_append(snapshot)

    # -- internals (call with lock unless noted) ---------------------------

    def _finish_locked(self, job: Job, state: JobState, error: JobError | None = None) -> None:
        job.state = state
        job.error = error
        job.finished_at = datetime.now(UTC)
        self._cancel_requested.discard(job.id)

    def _update_progress(self, job_id: str, progress: JobProgress) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.progress = progress

    def _is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancel_requested

    def _journal_append(self, job: Job) -> None:
        """Append a full job snapshot (last line per id wins on replay). Lock-free:
        takes a snapshot argument so file I/O never happens under the lock."""
        path = self._journal_path
        if path is None:
            return
        try:
            with open(path, "a") as f:
                f.write(json.dumps(job.model_dump(mode="json")) + "\n")
        except OSError:
            log.exception("Failed to append job journal %s", path)

    def _replay_journal_locked(self) -> list[Job]:
        """Load journal history; mark non-terminal entries failed. Returns the
        jobs that were rewritten (caller journals them outside the lock)."""
        path = self._journal_path
        if path is None or not path.exists():
            return []
        latest: dict[str, Job] = {}
        order: list[str] = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        job = Job.model_validate(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        log.warning("Skipping malformed journal line in %s", path)
                        continue
                    if job.id not in latest:
                        order.append(job.id)
                    latest[job.id] = job
        except OSError:
            log.exception("Failed to read job journal %s", path)
            return []

        stale: list[Job] = []
        for job_id in order:
            job = latest[job_id]
            if job.state not in TERMINAL_JOB_STATES:
                job.state = JobState.FAILED
                job.error = JobError(
                    code="server_restarted",
                    message="Server restarted while this job was active",
                )
                job.finished_at = datetime.now(UTC)
                stale.append(job.model_copy(deep=True))
            self._jobs[job_id] = job
            self._order.append(job_id)
        return stale
