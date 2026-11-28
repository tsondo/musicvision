"""Tests for assembly.concatenator.sync_warnings (structured rough-cut timing warnings)."""

from __future__ import annotations

from musicvision.assembly.concatenator import sync_warnings
from musicvision.models import Scene, SceneList

FPS = 25


def _scenes(*spans: tuple[float, float]) -> SceneList:
    return SceneList(scenes=[Scene(id=f"s{i}", order=i, time_start=a, time_end=b) for i, (a, b) in enumerate(spans)])


def test_within_one_frame_is_clean():
    assert sync_warnings(_scenes((0, 4), (4, 8)), 8.0 + 0.5 / FPS, FPS) == []


def test_gaps_explain_shortfall():
    scenes = _scenes((0, 4), (5, 8), (8, 10))  # 1s gap after s0
    (w,) = sync_warnings(scenes, 9.0, FPS)
    assert w["code"] == "av_drift"
    assert w["detail"]["drift_seconds"] == -1.0
    assert w["detail"]["drift_frames"] == -25
    assert w["detail"]["scene_gaps"] == [{"after": "s0", "before": "s1", "seconds": 1.0}]
    assert "gap" in w["message"] and "cut off" in w["message"]


def test_longer_video_without_gaps():
    (w,) = sync_warnings(_scenes((0, 4)), 5.0, FPS)
    assert w["detail"]["drift_seconds"] == 1.0
    assert w["detail"]["scene_gaps"] == []
    assert "longer" in w["message"]
