"""
Tests for the job-shaped pipeline endpoints (AGENT_INTERFACE_SPEC.md P2).

The /api/pipeline/* and per-scene regenerate/upscale endpoints now submit
jobs to the JobManager instead of blocking. These tests run the real
handlers end to end with fake engines: no GPU, no torch, no network.
"""

from __future__ import annotations

import threading
import time

import pytest

from musicvision.jobs import JobManager
from musicvision.models import JobProgress, Scene, SceneList

TICK = 0.005


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(TICK)
    pytest.fail("timed out waiting for condition")


def wait_terminal(c, job_id, timeout=5.0):
    result = {}

    def done():
        job = c.get(f"/api/jobs/{job_id}").json()
        result["job"] = job
        return job["state"] in ("succeeded", "failed", "cancelled")

    wait_for(done, timeout)
    return result["job"]


class FakeImageEngine:
    """Records calls; optionally blocks on a gate per generate()."""

    def __init__(self, gate: threading.Event | None = None):
        self.gate = gate
        self.generated: list[str] = []
        self.loaded = False
        self.unloaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.unloaded = True

    def generate(self, prompt, width, height, output_path, **kwargs):
        if self.gate is not None:
            self.gate.wait(timeout=5)
            self.gate.clear()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")
        self.generated.append(output_path.stem)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import musicvision.api.app as app_module
    import musicvision.utils.gpu as gpu

    fresh = JobManager()
    app_module.register_job_handlers(fresh)
    monkeypatch.setattr(app_module, "job_manager", fresh)
    monkeypatch.setattr(app_module, "_project", None)
    monkeypatch.setattr(gpu, "detect_devices", lambda: None)

    with TestClient(app_module.app) as c:
        resp = c.post("/api/projects/create", json={"name": "Job Endpoints", "directory": str(tmp_path / "proj")})
        assert resp.status_code == 200
        proj = app_module._project
        proj.scenes = SceneList(scenes=[
            Scene(id=f"scene_00{i}", order=i, time_start=i * 2.0, time_end=i * 2.0 + 2.0, image_prompt=f"shot {i}")
            for i in range(1, 4)
        ])
        proj.save_scenes()
        yield c, fresh, proj
    fresh.shutdown()


@pytest.fixture
def fake_image_engine(monkeypatch):
    import musicvision.imaging as imaging

    holder: dict = {}

    def factory(gate=None):
        engine = FakeImageEngine(gate)
        holder["engine"] = engine
        monkeypatch.setattr(imaging, "create_engine", lambda config, device_map: engine)
        return engine

    return factory


# ---------------------------------------------------------------------------
# Submission shape
# ---------------------------------------------------------------------------

class TestSubmission:
    def test_pipeline_endpoint_returns_queued_job(self, client, fake_image_engine):
        c, _, _ = client
        fake_image_engine()
        resp = c.post("/api/pipeline/generate-images", json={"scene_ids": ["scene_001"]})
        assert resp.status_code == 202
        job = resp.json()
        assert job["kind"] == "generate_images"
        assert job["params"]["scene_ids"] == ["scene_001"]
        assert job["id"].startswith("job_")

    def test_unknown_scene_is_structured_404_at_submit(self, client):
        c, _, _ = client
        resp = c.post("/api/pipeline/generate-images", json={"scene_ids": ["scene_999"]})
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "scene_not_found"
        assert resp.json()["error"]["detail"]["scene_ids"] == ["scene_999"]

        resp = c.post("/api/scenes/scene_999/regenerate-image", json={})
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "scene_not_found"

    def test_per_scene_job_is_scoped_to_its_scene(self, client, fake_image_engine):
        c, _, _ = client
        gate = threading.Event()
        fake_image_engine(gate)
        j1 = c.post("/api/scenes/scene_001/regenerate-image", json={}).json()
        assert j1["params"]["scene_id"] == "scene_001"
        assert j1["params"]["scene_ids"] == ["scene_001"]
        # A different scene can queue behind it; the same scene conflicts.
        assert c.post("/api/scenes/scene_002/regenerate-image", json={}).status_code == 202
        dup = c.post("/api/scenes/scene_001/regenerate-image", json={})
        assert dup.status_code == 409
        assert dup.json()["error"]["code"] == "job_already_active"
        for _ in range(4):
            gate.set()
            time.sleep(0.02)

    def test_generic_submit_validates_params(self, client):
        c, _, _ = client
        resp = c.post("/api/jobs", json={"kind": "regenerate_image", "params": {}})
        assert resp.status_code == 400
        body = resp.json()["error"]
        assert body["code"] == "invalid_job_params"
        assert any("scene_id" in e["loc"] for e in body["detail"]["errors"])

        resp = c.post("/api/jobs", json={"kind": "generate_images", "params": {"scene_ids": "scene_001"}})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "invalid_job_params"

    def test_generic_submit_matches_endpoint(self, client, fake_image_engine):
        c, _, _ = client
        fake_image_engine()
        resp = c.post("/api/jobs", json={"kind": "regenerate_image", "params": {"scene_id": "scene_002"}})
        assert resp.status_code == 202
        job = wait_terminal(c, resp.json()["id"])
        assert job["state"] == "succeeded"
        assert job["result"]["scene"]["id"] == "scene_002"


# ---------------------------------------------------------------------------
# Real handlers with fake engines
# ---------------------------------------------------------------------------

class TestHandlers:
    def test_generate_images_reports_per_scene_progress(self, client, fake_image_engine):
        c, _, proj = client
        engine = fake_image_engine()
        job = c.post("/api/pipeline/generate-images", json={}).json()
        final = wait_terminal(c, job["id"])

        assert final["state"] == "succeeded"
        assert final["result"]["generated"] == ["scene_001", "scene_002", "scene_003"]
        assert final["result"]["failed"] == []
        assert final["progress"]["current"] == 3
        assert final["progress"]["total"] == 3
        assert engine.loaded and engine.unloaded
        assert all(s.reference_image == f"images/{s.id}.png" for s in proj.scenes.scenes)

    def test_cancel_stops_between_scenes_and_unloads(self, client, fake_image_engine):
        c, _, proj = client
        gate = threading.Event()
        engine = fake_image_engine(gate)
        job = c.post("/api/pipeline/generate-images", json={}).json()

        wait_for(lambda: c.get(f"/api/jobs/{job['id']}").json()["progress"].get("scene_id") == "scene_001")
        assert c.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
        gate.set()  # let scene_001 finish; the handler must stop before scene_002

        final = wait_terminal(c, job["id"])
        assert final["state"] == "cancelled"
        assert final["result"]["partial"] is True
        assert engine.generated == ["scene_001"]
        assert engine.unloaded
        assert proj.scenes.get_scene("scene_001").reference_image
        assert proj.scenes.get_scene("scene_002").reference_image is None

    def test_regenerate_image_returns_updated_scene(self, client, fake_image_engine):
        c, _, _ = client
        fake_image_engine()
        job = c.post("/api/scenes/scene_003/regenerate-image", json={"seed": 7}).json()
        final = wait_terminal(c, job["id"])
        assert final["state"] == "succeeded"
        scene = final["result"]["scene"]
        assert scene["id"] == "scene_003"
        assert scene["reference_image"] == "images/scene_003.png"
        assert scene["image_status"] == "pending"
        assert final["progress"] == {
            "current": 1, "total": 1, "unit": "scenes",
            "message": "Generated image for scene_003", "scene_id": "scene_003",
        }

    def test_generate_videos_checks_prerequisites_at_run_time(self, client):
        c, _, _ = client
        # No reference images yet: submit succeeds (an image job could be
        # queued ahead of it), the job fails with a typed error when it runs.
        resp = c.post("/api/pipeline/generate-videos", json={})
        assert resp.status_code == 202
        final = wait_terminal(c, resp.json()["id"])
        assert final["state"] == "failed"
        assert final["error"]["code"] == "missing_prerequisite"
        assert final["error"]["detail"]["scene_ids"] == ["scene_001", "scene_002", "scene_003"]

    def test_generate_videos_runs_per_scene_with_fake_engine(self, client, monkeypatch):
        from types import SimpleNamespace

        import musicvision.video as video

        c, _, proj = client
        song = proj.paths.root / "input" / "song.wav"
        song.parent.mkdir(parents=True, exist_ok=True)
        song.write_bytes(b"RIFF")
        proj.config.song.audio_file = "input/song.wav"
        for scene in proj.scenes.scenes:
            scene.reference_image = f"images/{scene.id}.png"
            scene.video_prompt = f"motion for {scene.id}"

        rendered: list[str] = []

        class FakeVideoEngine:
            unloaded = False

            def load(self):
                pass

            def unload(self):
                FakeVideoEngine.unloaded = True

            def generate_scene(self, *, output_dir, scene_id, **kwargs):
                rendered.append(scene_id)
                clip = output_dir / f"{scene_id}.mp4"
                clip.parent.mkdir(parents=True, exist_ok=True)
                clip.write_bytes(b"mp4")
                return [SimpleNamespace(video_path=clip, generated_audio_path=None)]

        monkeypatch.setattr(video, "create_video_engine", lambda *a, **kw: FakeVideoEngine())
        monkeypatch.setattr("musicvision.utils.video.update_scene_resolution", lambda *a, **kw: None)

        job = c.post("/api/pipeline/generate-videos", json={"render_mode": "preview"}).json()
        final = wait_terminal(c, job["id"])
        assert final["state"] == "succeeded", final["error"]
        assert final["result"]["generated"] == ["scene_001", "scene_002", "scene_003"]
        assert final["progress"]["current"] == 3 and final["progress"]["total"] == 3
        assert rendered == ["scene_001", "scene_002", "scene_003"]
        assert FakeVideoEngine.unloaded
        assert proj.scenes.get_scene("scene_002").video_clip == "clips/scene_002.mp4"

    def test_descriptions_batch_reports_progress(self, client, monkeypatch):
        import musicvision.imaging.prompt_generator as pg

        c, _, proj = client
        proj.scenes.get_scene("scene_002").image_prompt = None
        monkeypatch.setattr(pg, "generate_image_prompts_batch", lambda scenes, config: [f"new {s.id}" for s in scenes])
        job = c.post("/api/pipeline/generate-descriptions", json={}).json()
        final = wait_terminal(c, job["id"])
        assert final["state"] == "succeeded"
        assert final["result"] == {"status": "complete", "generated": ["scene_002"], "failed": [], "total": 1}
        assert final["progress"]["current"] == 1 and final["progress"]["total"] == 1
        assert proj.scenes.get_scene("scene_002").image_prompt == "new scene_002"

    def test_assemble_without_audio_fails_typed(self, client):
        c, _, _ = client
        resp = c.post("/api/pipeline/assemble", json={})
        assert resp.status_code == 202
        final = wait_terminal(c, resp.json()["id"])
        assert final["state"] == "failed"
        assert final["error"]["code"] == "missing_prerequisite"
        assert final["error"]["detail"]["missing"] == "audio_file"

    def test_upscale_scene_without_clip_fails_typed(self, client):
        c, _, _ = client
        resp = c.post("/api/scenes/scene_001/upscale", json={})
        assert resp.status_code == 202
        final = wait_terminal(c, resp.json()["id"])
        assert final["state"] == "failed"
        assert final["error"]["code"] == "missing_prerequisite"

    def test_oom_becomes_vram_exhausted(self, client, monkeypatch):
        import musicvision.imaging as imaging

        class OomEngine(FakeImageEngine):
            def generate(self, *a, **kw):
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        monkeypatch.setattr(imaging, "create_engine", lambda config, device_map: OomEngine())
        c, _, _ = client
        job = c.post("/api/scenes/scene_001/regenerate-image", json={}).json()
        final = wait_terminal(c, job["id"])
        assert final["state"] == "failed"
        assert final["error"]["code"] == "vram_exhausted"


# ---------------------------------------------------------------------------
# Core: upscale_clips progress + cooperative cancel
# ---------------------------------------------------------------------------

class TestUpscaleCallbacks:
    def _setup(self, tmp_path, monkeypatch):
        import musicvision.upscaling.pipeline as up
        from musicvision.models import UpscalerConfig, UpscalerType, VideoEngineType
        from musicvision.utils.paths import ProjectPaths

        calls: list[str] = []

        class FakeUpscaler:
            def load(self):
                pass

            def unload(self):
                calls.append("unload")

        monkeypatch.setattr(up, "create_upscale_engine", lambda *a, **kw: FakeUpscaler())
        monkeypatch.setattr(up, "_upscale_scene", lambda scene, *a, **kw: calls.append(scene.id))
        monkeypatch.setattr("musicvision.utils.video.update_scene_resolution", lambda *a, **kw: None)

        scenes = SceneList(scenes=[
            Scene(id=f"s{i}", order=i, time_start=i, time_end=i + 1, video_clip=f"clips/s{i}.mp4")
            for i in range(3)
        ])
        config = UpscalerConfig(upscaler_override=UpscalerType.REAL_ESRGAN)
        return up, scenes, ProjectPaths(tmp_path), config, VideoEngineType.HUMO, calls

    def test_progress_per_scene(self, tmp_path, monkeypatch):
        up, scenes, paths, config, engine, calls = self._setup(tmp_path, monkeypatch)
        seen: list[JobProgress] = []
        result = up.upscale_clips(scenes, paths, config, engine, on_progress=seen.append)
        assert result == {"upscaled": ["s0", "s1", "s2"], "failed": []}
        assert [p.current for p in seen if p.message.startswith("Upscaled")] == [1, 2, 3]
        assert all(p.total == 3 for p in seen)

    def test_cancel_between_scenes(self, tmp_path, monkeypatch):
        up, scenes, paths, config, engine, calls = self._setup(tmp_path, monkeypatch)
        result = up.upscale_clips(
            scenes, paths, config, engine, should_cancel=lambda: len([c for c in calls if c != "unload"]) >= 1,
        )
        assert result["upscaled"] == ["s0"]
        assert result["cancelled"] is True
        assert calls[-1] == "unload"
