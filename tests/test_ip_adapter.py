"""Phase 3 IP-Adapter tests — mock-pipe pattern, no GPU, no weights, no network."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch required for image engine tests")

from musicvision.imaging.flux_engine import FluxEngine
from musicvision.imaging.zimage_engine import ZImageEngine
from musicvision.models import (
    ImageGenConfig,
    ImageModel,
    IPAdapterConfig,
    ProjectConfig,
)
from musicvision.utils.gpu import DeviceMap

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def gpu_device_map():
    """Fake GPU device map (no actual GPU needed)."""
    return DeviceMap(
        dit_device=torch.device("cuda:0"),
        encoder_device=torch.device("cuda:1"),
        vae_device=torch.device("cuda:1"),
        offload_device=torch.device("cpu"),
    )


def _mock_pipe():
    """Create a mock diffusers pipeline."""
    pipe = MagicMock()
    mock_image = MagicMock()
    pipe.return_value.images = [mock_image]
    return pipe, mock_image


def _create_test_image(path: Path) -> Path:
    """Write a tiny real PNG so PIL can open it."""
    from PIL import Image

    Image.new("RGB", (8, 8), color=(200, 30, 30)).save(path)
    return path


def _create_test_embedding(path: Path) -> Path:
    """Write a tiny real tensor file loadable via torch.load(weights_only=True)."""
    torch.save(torch.zeros(4), path)
    return path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestIPAdapterConfig:
    def test_defaults_select_xlabs_v2(self):
        cfg = IPAdapterConfig()
        assert cfg.enabled is False
        assert cfg.model_repo == "XLabs-AI/flux-ip-adapter-v2"
        assert cfg.weight_name == "ip_adapter.safetensors"
        assert cfg.image_encoder_repo == "openai/clip-vit-large-patch14"
        assert cfg.default_scale == 0.6

    def test_v1_is_pure_config_swap(self):
        """XLabs v1 needs only a repo change — same loader, same encoder."""
        cfg = IPAdapterConfig(model_repo="XLabs-AI/flux-ip-adapter")
        assert cfg.weight_name == "ip_adapter.safetensors"
        assert cfg.image_encoder_repo == "openai/clip-vit-large-patch14"

    def test_image_gen_config_has_ip_adapter(self):
        cfg = ImageGenConfig()
        assert isinstance(cfg.ip_adapter, IPAdapterConfig)
        assert cfg.ip_adapter.enabled is False

    def test_project_yaml_without_ip_adapter_key_loads(self, tmp_path):
        """Backward compat: pre-Phase-3 project.yaml has no ip_adapter key."""
        config = ProjectConfig.model_validate({
            "name": "Old Project",
            "image_gen": {"model": "flux-dev", "steps": 28},
        })
        assert config.image_gen.ip_adapter.enabled is False

        # And roundtrips with the new key present
        path = tmp_path / "project.yaml"
        config.image_gen.ip_adapter.enabled = True
        config.save(path)
        loaded = ProjectConfig.load(path)
        assert loaded.image_gen.ip_adapter.enabled is True
        assert loaded.image_gen.ip_adapter.model_repo == "XLabs-AI/flux-ip-adapter-v2"


# ---------------------------------------------------------------------------
# FluxEngine generation with IP-Adapter (mock pipe)
# ---------------------------------------------------------------------------


class TestFluxIPAdapter:
    @patch("musicvision.imaging.flux_engine.clear_vram")
    def test_generate_with_ip_adapter_images(self, mock_clear, tmp_path, gpu_device_map):
        cfg = ImageGenConfig(ip_adapter=IPAdapterConfig(enabled=True))
        engine = FluxEngine(cfg, gpu_device_map)
        mock_pipe, mock_image = _mock_pipe()
        engine._pipe = mock_pipe
        engine._ip_adapter_loaded = True

        ref_img = _create_test_image(tmp_path / "ref.png")
        output = tmp_path / "out.png"
        engine.generate(
            "test prompt", output_path=output,
            ip_adapter_images=[ref_img],
            ip_adapter_scales=[0.7],
        )

        call_kwargs = mock_pipe.call_args[1]
        assert "ip_adapter_image" in call_kwargs
        assert "ip_adapter_image_embeds" not in call_kwargs
        mock_pipe.set_ip_adapter_scale.assert_called_once_with(0.7)

    @patch("musicvision.imaging.flux_engine.clear_vram")
    def test_generate_without_ip_adapter_no_kwargs(self, mock_clear, tmp_path, gpu_device_map):
        """When no IP-Adapter images provided, no IPA kwargs passed."""
        cfg = ImageGenConfig()
        engine = FluxEngine(cfg, gpu_device_map)
        mock_pipe, mock_image = _mock_pipe()
        engine._pipe = mock_pipe

        output = tmp_path / "out.png"
        engine.generate("test", output_path=output)

        call_kwargs = mock_pipe.call_args[1]
        assert "ip_adapter_image" not in call_kwargs
        assert "ip_adapter_image_embeds" not in call_kwargs
        mock_pipe.set_ip_adapter_scale.assert_not_called()
        mock_pipe.load_ip_adapter.assert_not_called()

    @patch("musicvision.imaging.flux_engine.clear_vram")
    def test_default_scale_when_none_given(self, mock_clear, tmp_path, gpu_device_map):
        cfg = ImageGenConfig(ip_adapter=IPAdapterConfig(enabled=True))
        engine = FluxEngine(cfg, gpu_device_map)
        mock_pipe, _ = _mock_pipe()
        engine._pipe = mock_pipe
        engine._ip_adapter_loaded = True

        ref_img = _create_test_image(tmp_path / "ref.png")
        engine.generate("test", output_path=tmp_path / "out.png", ip_adapter_images=[ref_img])

        mock_pipe.set_ip_adapter_scale.assert_called_once_with(0.6)

    @patch("musicvision.imaging.flux_engine.clear_vram")
    def test_multi_ipa_passes_lists(self, mock_clear, tmp_path, gpu_device_map):
        """Two conditioning assets → list of images + list of scales."""
        cfg = ImageGenConfig(ip_adapter=IPAdapterConfig(enabled=True))
        engine = FluxEngine(cfg, gpu_device_map)
        mock_pipe, _ = _mock_pipe()
        engine._pipe = mock_pipe
        engine._ip_adapter_loaded = True

        a = _create_test_image(tmp_path / "a.png")
        b = _create_test_image(tmp_path / "b.png")
        engine.generate(
            "test", output_path=tmp_path / "out.png",
            ip_adapter_images=[a, b],
            ip_adapter_scales=[0.5, 0.7],
        )

        call_kwargs = mock_pipe.call_args[1]
        assert isinstance(call_kwargs["ip_adapter_image"], list)
        assert len(call_kwargs["ip_adapter_image"]) == 2
        mock_pipe.set_ip_adapter_scale.assert_called_once_with([0.5, 0.7])

    @patch("musicvision.imaging.flux_engine.clear_vram")
    def test_embeddings_preferred_over_images(self, mock_clear, tmp_path, gpu_device_map):
        cfg = ImageGenConfig(ip_adapter=IPAdapterConfig(enabled=True))
        engine = FluxEngine(cfg, gpu_device_map)
        mock_pipe, _ = _mock_pipe()
        engine._pipe = mock_pipe
        engine._ip_adapter_loaded = True

        ref_img = _create_test_image(tmp_path / "ref.png")
        emb = _create_test_embedding(tmp_path / "singer.ipadpt")
        engine.generate(
            "test", output_path=tmp_path / "out.png",
            ip_adapter_images=[ref_img],
            ip_adapter_scales=[0.6],
            ip_adapter_embeddings=[emb],
        )

        call_kwargs = mock_pipe.call_args[1]
        assert "ip_adapter_image_embeds" in call_kwargs
        assert "ip_adapter_image" not in call_kwargs
        assert len(call_kwargs["ip_adapter_image_embeds"]) == 1
        assert torch.equal(call_kwargs["ip_adapter_image_embeds"][0], torch.zeros(4))


# ---------------------------------------------------------------------------
# Lazy load + lifecycle
# ---------------------------------------------------------------------------


class TestIPAdapterLifecycle:
    @patch("musicvision.imaging.flux_engine.clear_vram")
    def test_lazy_load_on_first_use(self, mock_clear, tmp_path, gpu_device_map):
        """Adapter not loaded in load() but IPA inputs passed → lazy load (no offload)."""
        cfg = ImageGenConfig()  # ip_adapter.enabled=False
        engine = FluxEngine(cfg, gpu_device_map)
        mock_pipe, _ = _mock_pipe()
        engine._pipe = mock_pipe
        assert not engine._ip_adapter_loaded
        assert not engine._offload_active

        ref_img = _create_test_image(tmp_path / "ref.png")
        engine.generate("test", output_path=tmp_path / "out.png", ip_adapter_images=[ref_img])

        mock_pipe.load_ip_adapter.assert_called_once()
        _, load_kwargs = mock_pipe.load_ip_adapter.call_args
        assert load_kwargs["weight_name"] == "ip_adapter.safetensors"
        assert engine._ip_adapter_loaded
        # image encoder placed with the other encoders on the secondary GPU
        mock_pipe.image_encoder.to.assert_called_with(gpu_device_map.encoder_device)
        assert "ip_adapter_image" in mock_pipe.call_args[1]

    @patch("musicvision.imaging.flux_engine.clear_vram")
    def test_lazy_load_blocked_under_offload(self, mock_clear, tmp_path, gpu_device_map, caplog):
        """cpu-offload active + adapter not loaded → warn and skip, never load."""
        cfg = ImageGenConfig()
        engine = FluxEngine(cfg, gpu_device_map)
        mock_pipe, _ = _mock_pipe()
        engine._pipe = mock_pipe
        engine._offload_active = True

        ref_img = _create_test_image(tmp_path / "ref.png")
        with caplog.at_level("WARNING"):
            engine.generate("test", output_path=tmp_path / "out.png", ip_adapter_images=[ref_img])

        mock_pipe.load_ip_adapter.assert_not_called()
        assert not engine._ip_adapter_loaded
        assert "cpu-offload" in caplog.text
        call_kwargs = mock_pipe.call_args[1]
        assert "ip_adapter_image" not in call_kwargs

    @patch("musicvision.imaging.flux_engine.clear_vram")
    def test_unload_resets_ip_adapter_state(self, mock_clear, gpu_device_map):
        engine = FluxEngine(ImageGenConfig(), gpu_device_map)
        mock_pipe, _ = _mock_pipe()
        engine._pipe = mock_pipe
        engine._ip_adapter_loaded = True
        engine._offload_active = True

        engine.unload()
        assert engine._pipe is None
        assert not engine._ip_adapter_loaded
        assert not engine._offload_active


# ---------------------------------------------------------------------------
# load() ordering — IP-Adapter strictly before cpu-offload / placement
# ---------------------------------------------------------------------------


class TestLoadOrdering:
    def _tracked_pipe(self, calls: list):
        pipe = MagicMock()
        pipe.load_ip_adapter.side_effect = lambda *a, **k: calls.append("ip_adapter")
        pipe.enable_model_cpu_offload.side_effect = lambda *a, **k: calls.append("offload")
        pipe.enable_sequential_cpu_offload.side_effect = lambda *a, **k: calls.append("seq_offload")
        return pipe

    @patch("musicvision.imaging.flux_engine._free_vram_gb", return_value=20.0)
    def test_ip_adapter_loads_before_model_cpu_offload(self, mock_vram, gpu_device_map):
        diffusers = pytest.importorskip("diffusers")
        calls: list = []
        pipe = self._tracked_pipe(calls)
        cfg = ImageGenConfig(ip_adapter=IPAdapterConfig(enabled=True))
        engine = FluxEngine(cfg, gpu_device_map)

        with patch.object(diffusers.FluxPipeline, "from_pretrained", return_value=pipe):
            engine.load()

        assert calls == ["ip_adapter", "offload"]  # 20 GB free → Tier B model-offload
        assert engine._ip_adapter_loaded
        assert engine._offload_active

    @patch("musicvision.imaging.flux_engine._free_vram_gb", return_value=30.0)
    def test_split_placement_parks_burst_encoders(self, mock_vram, gpu_device_map):
        """Split v2: execution on the DiT GPU (transformer + VAE); T5 and the
        IP-Adapter image encoder are CPU-parked and raised per encode burst."""
        diffusers = pytest.importorskip("diffusers")
        calls: list = []
        pipe = self._tracked_pipe(calls)
        cfg = ImageGenConfig(ip_adapter=IPAdapterConfig(enabled=True))
        engine = FluxEngine(cfg, gpu_device_map)

        with patch.object(diffusers.FluxPipeline, "from_pretrained", return_value=pipe):
            engine.load()

        # 30 GB free + two devices → split placement, no offload at all
        assert calls == ["ip_adapter"]
        assert not engine._offload_active
        pipe.transformer.to.assert_called_once_with(gpu_device_map.dit_device)
        pipe.vae.to.assert_called_once_with(gpu_device_map.dit_device)
        pipe.text_encoder.to.assert_called_once_with(gpu_device_map.encoder_device)
        pipe.text_encoder_2.to.assert_called_once_with("cpu")
        pipe.image_encoder.to.assert_called_once_with("cpu")

    @patch("musicvision.imaging.flux_engine._free_vram_gb", return_value=20.0)
    def test_disabled_ip_adapter_never_loads(self, mock_vram, gpu_device_map):
        diffusers = pytest.importorskip("diffusers")
        calls: list = []
        pipe = self._tracked_pipe(calls)
        engine = FluxEngine(ImageGenConfig(), gpu_device_map)

        with patch.object(diffusers.FluxPipeline, "from_pretrained", return_value=pipe):
            engine.load()

        assert calls == ["offload"]
        assert not engine._ip_adapter_loaded


# ---------------------------------------------------------------------------
# Z-Image — accept and ignore
# ---------------------------------------------------------------------------


class TestZImageIPAdapter:
    @patch("musicvision.imaging.zimage_engine.clear_vram")
    def test_zimage_ignores_ip_adapter(self, mock_clear, tmp_path, gpu_device_map, caplog):
        """Z-Image logs a warning but doesn't crash when IPA params passed."""
        cfg = ImageGenConfig(model=ImageModel.ZIMAGE)
        engine = ZImageEngine(cfg, gpu_device_map)
        mock_pipe, mock_image = _mock_pipe()
        engine._pipe = mock_pipe

        ref_img = _create_test_image(tmp_path / "ref.png")
        output = tmp_path / "out.png"
        with caplog.at_level("WARNING"):
            result = engine.generate("test", output_path=output, ip_adapter_images=[ref_img])

        assert result.prompt == "test"
        assert "IP-Adapter not supported on Z-Image" in caplog.text
        call_kwargs = mock_pipe.call_args[1]
        assert "ip_adapter_image" not in call_kwargs

    @patch("musicvision.imaging.zimage_engine.clear_vram")
    def test_zimage_no_warning_without_ipa_inputs(self, mock_clear, tmp_path, gpu_device_map, caplog):
        cfg = ImageGenConfig(model=ImageModel.ZIMAGE)
        engine = ZImageEngine(cfg, gpu_device_map)
        mock_pipe, _ = _mock_pipe()
        engine._pipe = mock_pipe

        with caplog.at_level("WARNING"):
            engine.generate("test", output_path=tmp_path / "out.png")

        assert "IP-Adapter" not in caplog.text
