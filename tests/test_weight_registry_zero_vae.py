"""Official HuMo zero_vae files are part of the shared weights (download-weights fetches them)."""

from __future__ import annotations

from pathlib import Path

from musicvision.models import HumoTier
from musicvision.video import weight_registry as wr

ZERO_VAE = {"zero_vae_480p": "zero_vae_129frame.pt", "zero_vae_720p": "zero_vae_720p_161frame.pt"}


def test_specs_point_at_official_files():
    for key, filename in ZERO_VAE.items():
        spec = wr.SHARED_SPECS[key]
        assert spec.repo_id == "bytedance-research/HuMo"
        assert spec.filename == filename


def test_weight_status_reports_zero_vae(tmp_path):
    status = wr.weight_status(HumoTier.FP8_SCALED, base_dir=tmp_path)
    assert status["zero_vae_480p"] is False and status["zero_vae_720p"] is False


def test_download_all_for_tier_fetches_zero_vae(tmp_path, monkeypatch):
    fetched: list[tuple[str, str]] = []

    def fake_hf_hub_download(repo_id, filename, local_dir, token=None, **kw):
        fetched.append((repo_id, filename))
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(wr, "download_dit", lambda tier, base_dir=None, hf_token=None: tmp_path / "dit")

    paths = wr.download_all_for_tier(HumoTier.FP8_SCALED, base_dir=tmp_path)
    for key, filename in ZERO_VAE.items():
        assert ("bytedance-research/HuMo", filename) in fetched
        assert paths[key] == wr.locate_shared(key, base_dir=tmp_path)
        assert paths[key].name == filename
