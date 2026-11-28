"""
Tests for the job manager (jobs.py) and the /api/jobs endpoints.

No GPU, no engines: handlers are fakes that loop over fake units. Worker
timing is bounded by wait_for() polling, not sleeps of fixed length.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from musicvision.jobs import (
    DuplicateJobError,
    JobFailureError,
    JobManager,
    JobNotCancellableError,
    JobNotFoundError,
    UnknownJobKindError,
)
from musicvision.models import JobKind, JobState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICK = 0.005


def wait_for(predicate, timeout=5.0):
    """Poll until predicate() is truthy; fail the test on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(TICK)
    pytest.fail("timed out waiting for condition")


def make_counting_handler(units=3, unit_time=0.0, record=None):
    """Handler that processes `units` fake scenes, reporting progress."""

    def handler(params, ctx):
        for i in range(units):
            ctx.raise_if_cancelled()
            if unit_time:
                time.sleep(unit_time)
            ctx.report(current=i + 1, total=units, message=f"unit {i + 1}/{units}")
            if record is not None:
                record.append(i + 1)
        return {"units_done": units}

    return handler


def make_gated_handler(gate: threading.Event):
    """Handler that stays running until *gate* is set: keeps a job active
    deterministically, with no sleeps for the test to wait out."""
    def handler(params, ctx):
        gate.wait(timeout=5)
        ctx.report(current=1, total=1)
        return {}
    return handler


@pytest.fixture
def manager(tmp_path):
    m = JobManager()
    m.attach_project(tmp_path)
    yield m
    m.shutdown()


# ---------------------------------------------------------------------------
# JobManager lifecycle
# ---------------------------------------------------------------------------

class TestJobLifecycle:
    def test_submit_runs_to_success(self, manager):
        manager.register(JobKind.GENERATE_IMAGES, make_counting_handler(units=3))
        job = manager.submit(JobKind.GENERATE_IMAGES)
        assert job.state == JobState.QUEUED
        assert job.id.startswith("job_")
        assert job.created_at is not None

        wait_for(lambda: manager.get(job.id).state == JobState.SUCCEEDED)
        done = manager.get(job.id)
        assert done.result == {"units_done": 3}
        assert done.progress.current == 3
        assert done.progress.total == 3
        assert done.started_at is not None
        assert done.finished_at is not None
        assert done.error is None

    def test_progress_is_observable_mid_run(self, manager):
        gate = threading.Event()

        def handler(params, ctx):
            ctx.report(current=1, total=2, message="halfway", scene_id="scene_001")
            gate.wait(timeout=5)
            ctx.report(current=2, total=2)
            return {}

        manager.register(JobKind.GENERATE_VIDEOS, handler)
        job = manager.submit(JobKind.GENERATE_VIDEOS)
        wait_for(lambda: manager.get(job.id).progress.current == 1)
        mid = manager.get(job.id)
        assert mid.state == JobState.RUNNING
        assert mid.progress.message == "halfway"
        assert mid.progress.scene_id == "scene_001"
        gate.set()
        wait_for(lambda: manager.get(job.id).state == JobState.SUCCEEDED)

    def test_jobs_run_serially_in_fifo_order(self, manager):
        order = []

        def make(tag):
            def handler(params, ctx):
                order.append(f"{tag}-start")
                time.sleep(0.02)
                order.append(f"{tag}-end")
                return {}

            return handler

        manager.register(JobKind.GENERATE_IMAGES, make("img"))
        manager.register(JobKind.UPSCALE, make("ups"))
        j1 = manager.submit(JobKind.GENERATE_IMAGES)
        j2 = manager.submit(JobKind.UPSCALE)
        wait_for(lambda: manager.get(j2.id).state == JobState.SUCCEEDED)
        assert manager.get(j1.id).state == JobState.SUCCEEDED
        assert order == ["img-start", "img-end", "ups-start", "ups-end"]

    def test_typed_failure(self, manager):
        def handler(params, ctx):
            raise JobFailureError("vram_exhausted", "OOM on secondary GPU", {"gpu": 1})

        manager.register(JobKind.GENERATE_VIDEOS, handler)
        job = manager.submit(JobKind.GENERATE_VIDEOS)
        wait_for(lambda: manager.get(job.id).state == JobState.FAILED)
        failed = manager.get(job.id)
        assert failed.error.code == "vram_exhausted"
        assert failed.error.detail == {"gpu": 1}

    def test_unexpected_exception_fails_job_and_worker_survives(self, manager):
        def bad(params, ctx):
            raise RuntimeError("boom")

        manager.register(JobKind.ASSEMBLE, bad)
        manager.register(JobKind.GENERATE_IMAGES, make_counting_handler(units=1))
        j1 = manager.submit(JobKind.ASSEMBLE)
        wait_for(lambda: manager.get(j1.id).state == JobState.FAILED)
        assert manager.get(j1.id).error.code == "unhandled_exception"
        assert "boom" in manager.get(j1.id).error.message
        # Worker thread must survive a crashing handler
        j2 = manager.submit(JobKind.GENERATE_IMAGES)
        wait_for(lambda: manager.get(j2.id).state == JobState.SUCCEEDED)

    def test_unknown_kind_rejected(self, manager):
        with pytest.raises(UnknownJobKindError) as exc:
            manager.submit(JobKind.INTAKE)
        assert exc.value.code == "invalid_job_params"

    def test_get_unknown_id(self, manager):
        with pytest.raises(JobNotFoundError):
            manager.get("job_nope")


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

class TestCancellation:
    def test_cancel_queued_job_is_immediate(self, manager):
        gate = threading.Event()

        def blocker(params, ctx):
            gate.wait(timeout=5)
            return {}

        manager.register(JobKind.GENERATE_IMAGES, blocker)
        manager.register(JobKind.UPSCALE, make_counting_handler())
        running = manager.submit(JobKind.GENERATE_IMAGES)
        wait_for(lambda: manager.get(running.id).state == JobState.RUNNING)
        queued = manager.submit(JobKind.UPSCALE)

        cancelled = manager.cancel(queued.id)
        assert cancelled.state == JobState.CANCELLED
        gate.set()
        wait_for(lambda: manager.get(running.id).state == JobState.SUCCEEDED)
        # The cancelled job never ran
        assert manager.get(queued.id).started_at is None

    def test_cancel_running_job_stops_at_unit_boundary(self, manager):
        started = threading.Event()
        release = threading.Event()

        def handler(params, ctx):
            for i in range(10):
                ctx.raise_if_cancelled()
                started.set()
                release.wait(timeout=5)
                release.clear()
                ctx.report(current=i + 1, total=10)
            return {"units_done": 10}

        manager.register(JobKind.GENERATE_VIDEOS, handler)
        job = manager.submit(JobKind.GENERATE_VIDEOS)
        assert started.wait(timeout=5)
        manager.cancel(job.id)      # sets the flag; handler sees it on next unit
        release.set()               # let current unit finish
        wait_for(lambda: manager.get(job.id).state == JobState.CANCELLED)
        done = manager.get(job.id)
        assert done.result["partial"] is True
        assert done.result["completed_units"] == 1  # finished exactly the in-flight unit

    def test_cancel_running_sets_flag_and_journals_it(self, manager, tmp_path):
        started, release = threading.Event(), threading.Event()

        def handler(params, ctx):
            started.set()
            release.wait(timeout=5)
            ctx.raise_if_cancelled()
            return {}

        manager.register(JobKind.GENERATE_VIDEOS, handler)
        job = manager.submit(JobKind.GENERATE_VIDEOS)
        assert started.wait(timeout=5)
        snap = manager.cancel(job.id)
        assert snap.state == JobState.RUNNING and snap.cancel_requested is True
        assert manager.get(job.id).cancel_requested is True
        journal = [json.loads(line) for line in (tmp_path / "jobs.jsonl").read_text().splitlines()]
        assert any(e["id"] == job.id and e["state"] == "running" and e["cancel_requested"] for e in journal)
        release.set()
        wait_for(lambda: manager.get(job.id).state == JobState.CANCELLED)

    def test_cancel_during_last_unit_ends_succeeded_with_flag(self, manager):
        started, release = threading.Event(), threading.Event()

        def handler(params, ctx):  # single-unit job, like regenerate_image
            ctx.raise_if_cancelled()
            started.set()
            release.wait(timeout=5)
            ctx.report(current=1, total=1)
            return {"status": "complete"}

        manager.register(JobKind.REGENERATE_IMAGE, handler)
        job = manager.submit(JobKind.REGENERATE_IMAGE)
        assert started.wait(timeout=5)
        manager.cancel(job.id)
        release.set()
        wait_for(lambda: manager.get(job.id).is_terminal)
        done = manager.get(job.id)
        assert done.state == JobState.SUCCEEDED  # all work finished before a cancellation point
        assert done.cancel_requested is True
        assert done.result == {"status": "complete"}

    def test_cancelled_result_reports_partial_lists(self, manager):
        started, release = threading.Event(), threading.Event()

        def handler(params, ctx):
            generated, failed = [], []
            ctx.partial_result(generated=generated, failed=failed)
            for i, sid in enumerate(["s1", "s2", "s3", "s4"]):
                ctx.raise_if_cancelled()
                if sid == "s2":
                    failed.append({"scene_id": sid, "error": "boom"})
                else:
                    generated.append(sid)
                ctx.report(current=i + 1, total=4)
                if i == 1:
                    started.set()
                    release.wait(timeout=5)
            return {"generated": generated, "failed": failed}

        manager.register(JobKind.GENERATE_VIDEOS, handler)
        job = manager.submit(JobKind.GENERATE_VIDEOS)
        assert started.wait(timeout=5)
        manager.cancel(job.id)
        release.set()
        wait_for(lambda: manager.get(job.id).state == JobState.CANCELLED)
        result = manager.get(job.id).result
        assert result["partial"] is True
        assert result["completed_units"] == 2          # processed, successful or not
        assert result["generated"] == ["s1"]           # what actually succeeded
        assert result["failed"] == [{"scene_id": "s2", "error": "boom"}]

    def test_old_journal_entries_without_cancel_requested_load(self, tmp_path):
        from musicvision.models import Job

        legacy = {"id": "job_old", "kind": "generate_images", "params": {}, "state": "succeeded",
                  "progress": {"current": 1, "total": 1, "unit": "scenes", "message": "", "scene_id": None},
                  "error": None, "result": {}, "created_at": "2026-11-01T00:00:00Z",
                  "started_at": None, "finished_at": None}
        assert Job.model_validate(legacy).cancel_requested is False

    def test_cancel_terminal_job_rejected(self, manager):
        manager.register(JobKind.GENERATE_IMAGES, make_counting_handler(units=1))
        job = manager.submit(JobKind.GENERATE_IMAGES)
        wait_for(lambda: manager.get(job.id).state == JobState.SUCCEEDED)
        with pytest.raises(JobNotCancellableError):
            manager.cancel(job.id)


# ---------------------------------------------------------------------------
# Duplicate-scope guard
# ---------------------------------------------------------------------------

class TestDuplicateGuard:
    @pytest.fixture
    def gate(self):
        gate = threading.Event()
        yield gate
        gate.set()  # release running jobs before the manager shuts down

    @pytest.fixture
    def slow_manager(self, manager, gate):
        manager.register(JobKind.REGENERATE_IMAGE, make_gated_handler(gate))
        manager.register(JobKind.GENERATE_IMAGES, make_gated_handler(gate))
        return manager

    def test_same_kind_whole_project_conflicts(self, slow_manager):
        slow_manager.submit(JobKind.GENERATE_IMAGES)
        with pytest.raises(DuplicateJobError) as exc:
            slow_manager.submit(JobKind.GENERATE_IMAGES)
        assert exc.value.code == "job_already_active"

    def test_disjoint_scene_scopes_can_queue(self, slow_manager):
        j1 = slow_manager.submit(JobKind.REGENERATE_IMAGE, {"scene_ids": ["scene_001"]})
        j2 = slow_manager.submit(JobKind.REGENERATE_IMAGE, {"scene_ids": ["scene_002"]})
        assert j1.id != j2.id

    def test_overlapping_scene_scopes_conflict(self, slow_manager):
        slow_manager.submit(JobKind.REGENERATE_IMAGE, {"scene_ids": ["scene_001", "scene_002"]})
        with pytest.raises(DuplicateJobError):
            slow_manager.submit(JobKind.REGENERATE_IMAGE, {"scene_ids": ["scene_002"]})

    def test_whole_project_conflicts_with_scoped(self, slow_manager):
        slow_manager.submit(JobKind.GENERATE_IMAGES, {"scene_ids": ["scene_001"]})
        with pytest.raises(DuplicateJobError):
            slow_manager.submit(JobKind.GENERATE_IMAGES)  # whole-project overlaps any scope

    def test_different_kinds_never_conflict(self, slow_manager):
        slow_manager.submit(JobKind.GENERATE_IMAGES)
        slow_manager.submit(JobKind.REGENERATE_IMAGE, {"scene_ids": ["scene_001"]})

    def test_terminal_job_frees_the_scope(self, slow_manager, gate):
        j1 = slow_manager.submit(JobKind.GENERATE_IMAGES)
        gate.set()
        wait_for(lambda: slow_manager.get(j1.id).state == JobState.SUCCEEDED)
        slow_manager.submit(JobKind.GENERATE_IMAGES)  # no conflict after terminal


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

class TestJournal:
    def test_journal_records_state_transitions(self, tmp_path):
        m = JobManager()
        m.attach_project(tmp_path)
        try:
            m.register(JobKind.GENERATE_IMAGES, make_counting_handler(units=1))
            job = m.submit(JobKind.GENERATE_IMAGES)
            wait_for(lambda: m.get(job.id).state == JobState.SUCCEEDED)
        finally:
            m.shutdown()

        lines = [json.loads(line) for line in (tmp_path / "jobs.jsonl").read_text().splitlines()]
        states = [line["state"] for line in lines if line["id"] == job.id]
        assert states == ["queued", "running", "succeeded"]

    def test_restart_marks_nonterminal_jobs_failed(self, tmp_path):
        # Simulate a serve process that died mid-job: journal ends at RUNNING.
        m1 = JobManager()
        m1.attach_project(tmp_path)
        gate = threading.Event()
        try:
            m1.register(JobKind.GENERATE_VIDEOS, lambda p, ctx: gate.wait(timeout=5) and {})
            job = m1.submit(JobKind.GENERATE_VIDEOS)
            wait_for(lambda: m1.get(job.id).state == JobState.RUNNING)
        finally:
            gate.set()
            m1.shutdown()  # journal may end at RUNNING or terminal depending on timing

        # Force the mid-run shape regardless of shutdown timing
        running_snapshot = json.loads(
            [
                line
                for line in (tmp_path / "jobs.jsonl").read_text().splitlines()
                if json.loads(line)["state"] == "running"
            ][0]
        )
        (tmp_path / "jobs.jsonl").write_text(json.dumps(running_snapshot) + "\n")

        m2 = JobManager()
        m2.attach_project(tmp_path)
        try:
            revived = m2.get(job.id)
            assert revived.state == JobState.FAILED
            assert revived.error.code == "server_restarted"
            # The rewrite itself is journaled, so a third restart sees terminal state
            lines = (tmp_path / "jobs.jsonl").read_text().splitlines()
            assert json.loads(lines[-1])["state"] == "failed"
        finally:
            m2.shutdown()

    def test_history_visible_in_list_after_restart(self, tmp_path):
        m1 = JobManager()
        m1.attach_project(tmp_path)
        try:
            m1.register(JobKind.GENERATE_IMAGES, make_counting_handler(units=1))
            job = m1.submit(JobKind.GENERATE_IMAGES)
            wait_for(lambda: m1.get(job.id).state == JobState.SUCCEEDED)
        finally:
            m1.shutdown()

        m2 = JobManager()
        m2.attach_project(tmp_path)
        try:
            listed = m2.list()
            assert [j.id for j in listed] == [job.id]
            assert listed[0].state == JobState.SUCCEEDED
        finally:
            m2.shutdown()

    def test_malformed_journal_lines_are_skipped(self, tmp_path):
        (tmp_path / "jobs.jsonl").write_text("not json\n{\"also\": \"not a job\"}\n")
        m = JobManager()
        m.attach_project(tmp_path)
        try:
            assert m.list() == []
        finally:
            m.shutdown()


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import musicvision.api.app as app_module

    # Fresh manager per test — the module-level one is process-global
    fresh = JobManager()
    monkeypatch.setattr(app_module, "job_manager", fresh)
    monkeypatch.setattr(app_module, "_project", None)
    with TestClient(app_module.app) as c:
        # Open a real (empty) project so jobs have a journal home
        resp = c.post("/api/projects/create", json={"name": "Jobs Test", "directory": str(tmp_path / "proj")})
        assert resp.status_code == 200
        yield c, fresh
    fresh.shutdown()


class TestJobEndpoints:
    def test_submit_poll_roundtrip(self, client):
        c, manager = client
        manager.register(JobKind.GENERATE_IMAGES, make_counting_handler(units=2))
        resp = c.post("/api/jobs", json={"kind": "generate_images"})
        assert resp.status_code == 202
        job = resp.json()
        assert job["state"] == "queued"

        def done():
            return c.get(f"/api/jobs/{job['id']}").json()["state"] == "succeeded"

        wait_for(done)
        final = c.get(f"/api/jobs/{job['id']}").json()
        assert final["result"] == {"units_done": 2}
        assert final["progress"]["current"] == 2

    def test_submit_unregistered_kind_is_structured_error(self, client):
        c, _ = client
        resp = c.post("/api/jobs", json={"kind": "assemble"})
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "invalid_job_params"
        assert "assemble" in body["error"]["message"]

    def test_submit_without_project_is_structured_error(self, client):
        c, _ = client
        assert c.post("/api/projects/close").status_code == 200
        resp = c.post("/api/jobs", json={"kind": "generate_images"})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "project_not_open"

    def test_duplicate_submit_conflict(self, client):
        c, manager = client
        gate = threading.Event()
        manager.register(JobKind.GENERATE_IMAGES, make_gated_handler(gate))
        try:
            assert c.post("/api/jobs", json={"kind": "generate_images"}).status_code == 202
            resp = c.post("/api/jobs", json={"kind": "generate_images"})
            assert resp.status_code == 409
            assert resp.json()["error"]["code"] == "job_already_active"
            assert resp.json()["error"]["detail"]["job_id"].startswith("job_")
        finally:
            gate.set()

    def test_get_unknown_job_404(self, client):
        c, _ = client
        resp = c.get("/api/jobs/job_missing")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "job_not_found"

    def test_list_with_filters(self, client):
        c, manager = client
        manager.register(JobKind.GENERATE_IMAGES, make_counting_handler(units=1))
        manager.register(JobKind.UPSCALE, make_counting_handler(units=1))
        j1 = c.post("/api/jobs", json={"kind": "generate_images"}).json()
        j2 = c.post("/api/jobs", json={"kind": "upscale"}).json()

        def both_done():
            listed = c.get("/api/jobs", params={"state": "succeeded"}).json()
            return {j["id"] for j in listed} == {j1["id"], j2["id"]}

        wait_for(both_done)
        only_upscale = c.get("/api/jobs", params={"kind": "upscale"}).json()
        assert [j["id"] for j in only_upscale] == [j2["id"]]
        # newest first
        all_jobs = c.get("/api/jobs").json()
        assert [j["id"] for j in all_jobs] == [j2["id"], j1["id"]]

    def test_list_invalid_filter_is_structured_error(self, client):
        c, _ = client
        resp = c.get("/api/jobs", params={"state": "exploded"})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "invalid_job_params"

    def test_cancel_endpoint(self, client):
        c, manager = client
        gate = threading.Event()

        def handler(params, ctx):
            for i in range(5):
                ctx.raise_if_cancelled()
                gate.wait(timeout=5)
                gate.clear()
                ctx.report(current=i + 1, total=5)
            return {}

        manager.register(JobKind.GENERATE_VIDEOS, handler)
        job = c.post("/api/jobs", json={"kind": "generate_videos"}).json()

        def running():
            return c.get(f"/api/jobs/{job['id']}").json()["state"] == "running"

        wait_for(running)
        assert c.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
        gate.set()
        def cancelled():
            return c.get(f"/api/jobs/{job['id']}").json()["state"] == "cancelled"

        wait_for(cancelled)
        final = c.get(f"/api/jobs/{job['id']}").json()
        assert final["result"]["partial"] is True

    def test_cancel_terminal_conflict(self, client):
        c, manager = client
        manager.register(JobKind.GENERATE_IMAGES, make_counting_handler(units=1))
        job = c.post("/api/jobs", json={"kind": "generate_images"}).json()

        def done():
            return c.get(f"/api/jobs/{job['id']}").json()["state"] == "succeeded"

        wait_for(done)
        resp = c.post(f"/api/jobs/{job['id']}/cancel")
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "job_not_cancellable"

    def test_project_close_blocked_while_job_active(self, client):
        c, manager = client
        gate = threading.Event()
        manager.register(JobKind.GENERATE_VIDEOS, lambda p, ctx: gate.wait(timeout=5) and {})
        job = c.post("/api/jobs", json={"kind": "generate_videos"}).json()

        def running():
            return c.get(f"/api/jobs/{job['id']}").json()["state"] == "running"

        wait_for(running)
        resp = c.post("/api/projects/close")
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "project_busy"
        gate.set()

        def done():
            return c.get(f"/api/jobs/{job['id']}").json()["state"] == "succeeded"

        wait_for(done)
        assert c.post("/api/projects/close").status_code == 200
