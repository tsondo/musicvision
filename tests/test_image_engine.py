"""Tests for the model-agnostic image engine abstraction."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch required for image engine tests")

from musicvision.imaging.base import ImageEngine, ImageResult
from musicvision.imaging.factory import create_engine
from musicvision.imaging.flux_engine import FluxEngine, MODEL_IDS as FLUX_MODEL_IDS
from musicvision.imaging.zimage_engine import ZImageEngine, MODEL_IDS as ZIMAGE_MODEL_IDS
from musicvision.models import (
    FluxConfig,
    FluxModel,
    ImageGenConfig,
    ImageModel,
    ProjectConfig,
)
from musicvision.utils.gpu import DeviceMap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cpu_device_map():
    cpu = torch.device("cpu")
    return DeviceMap(
        dit_device=cpu,
        encoder_device=cpu,
        vae_device=cpu,
        offload_device=cpu,
    )


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


# ---------------------------------------------------------------------------
# Config / backward compat
# ---------------------------------------------------------------------------


class TestConfigCompat:
    def test_image_model_enum_has_all_variants(self):
        assert ImageModel.FLUX_DEV.value == "flux-dev"
        assert ImageModel.FLUX_SCHNELL.value == "flux-schnell"
        assert ImageModel.ZIMAGE.value == "z-image"
        assert ImageModel.ZIMAGE_TURBO.value == "z-image-turbo"

    def test_flux_model_alias(self):
        """FluxModel is an alias for ImageModel."""
        assert FluxModel is ImageModel

    def test_flux_config_alias(self):
        """FluxConfig is an alias for ImageGenConfig."""
        assert FluxConfig is ImageGenConfig

    def test_image_gen_config_defaults(self):
        cfg = ImageGenConfig()
        assert cfg.model == ImageModel.FLUX_DEV
        assert cfg.steps == 28
        assert cfg.guidance_scale == 3.5

    def test_project_config_flux_key_migration(self):
        """Old project.yaml files with 'flux' key still load correctly."""
        config = ProjectConfig.model_validate({
            "name": "Test",
            "flux": {"model": "flux-schnell", "steps": 4},
        })
        assert config.image_gen.model == ImageModel.FLUX_SCHNELL
        assert config.image_gen.steps == 4

    def test_project_config_image_gen_key(self):
        config = ProjectConfig.model_validate({
            "name": "Test",
            "image_gen": {"model": "z-image", "steps": 20},
        })
        assert config.image_gen.model == ImageModel.ZIMAGE

    def test_project_config_flux_property(self):
        """Deprecated .flux property returns image_gen."""
        config = ProjectConfig()
        assert config.flux is config.image_gen

    def test_project_config_roundtrip_with_image_gen(self, tmp_path):
        config = ProjectConfig(
            name="Z-Image Project",
            image_gen=ImageGenConfig(model=ImageModel.ZIMAGE_TURBO, steps=8),
        )
        path = tmp_path / "project.yaml"
        config.save(path)
        loaded = ProjectConfig.load(path)
        assert loaded.image_gen.model == ImageModel.ZIMAGE_TURBO
        assert loaded.image_gen.steps == 8


# ---------------------------------------------------------------------------
# Model ID resolution
# ---------------------------------------------------------------------------


class TestModelIDs:
    def test_flux_dev_model_id(self):
        assert FLUX_MODEL_IDS[ImageModel.FLUX_DEV] == "black-forest-labs/FLUX.1-dev"

    def test_flux_schnell_model_id(self):
        assert FLUX_MODEL_IDS[ImageModel.FLUX_SCHNELL] == "black-forest-labs/FLUX.1-schnell"

    def test_zimage_model_id(self):
        assert ZIMAGE_MODEL_IDS[ImageModel.ZIMAGE] == "Tongyi-MAI/Z-Image"

    def test_zimage_turbo_model_id(self):
        assert ZIMAGE_MODEL_IDS[ImageModel.ZIMAGE_TURBO] == "Tongyi-MAI/Z-Image-Turbo"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_creates_flux_engine_for_flux_dev(self, cpu_device_map):
        cfg = ImageGenConfig(model=ImageModel.FLUX_DEV)
        engine = create_engine(cfg, cpu_device_map)
        assert isinstance(engine, FluxEngine)

    def test_creates_flux_engine_for_flux_schnell(self, cpu_device_map):
        cfg = ImageGenConfig(model=ImageModel.FLUX_SCHNELL)
        engine = create_engine(cfg, cpu_device_map)
        assert isinstance(engine, FluxEngine)

    def test_creates_zimage_engine(self, cpu_device_map):
        cfg = ImageGenConfig(model=ImageModel.ZIMAGE)
        engine = create_engine(cfg, cpu_device_map)
        assert isinstance(engine, ZImageEngine)

    def test_creates_zimage_turbo_engine(self, cpu_device_map):
        cfg = ImageGenConfig(model=ImageModel.ZIMAGE_TURBO)
        engine = create_engine(cfg, cpu_device_map)
        assert isinstance(engine, ZImageEngine)

    def test_returns_image_engine_subclass(self, cpu_device_map):
        for model in ImageModel:
            engine = create_engine(ImageGenConfig(model=model), cpu_device_map)
            assert isinstance(engine, ImageEngine)


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------


class TestEngineLifecycle:
    def test_flux_not_loaded_initially(self, cpu_device_map):
        engine = FluxEngine(ImageGenConfig(), cpu_device_map)
        assert not engine.is_loaded

    def test_zimage_not_loaded_initially(self, cpu_device_map):
        engine = ZImageEngine(ImageGenConfig(model=ImageModel.ZIMAGE), cpu_device_map)
        assert not engine.is_loaded

    def test_generate_before_load_raises(self, cpu_device_map):
        engine = FluxEngine(ImageGenConfig(), cpu_device_map)
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.generate("test prompt", output_path=Path("/tmp/test.png"))

    def test_zimage_generate_before_load_raises(self, cpu_device_map):
        engine = ZImageEngine(ImageGenConfig(model=ImageModel.ZIMAGE), cpu_device_map)
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.generate("test prompt", output_path=Path("/tmp/test.png"))

    def test_unload_when_not_loaded_is_safe(self, cpu_device_map):
        engine = FluxEngine(ImageGenConfig(), cpu_device_map)
        with patch("musicvision.imaging.flux_engine.clear_vram"):
            engine.unload()  # should not raise

    def test_zimage_unload_when_not_loaded_is_safe(self, cpu_device_map):
        engine = ZImageEngine(ImageGenConfig(model=ImageModel.ZIMAGE), cpu_device_map)
        with patch("musicvision.imaging.zimage_engine.clear_vram"):
            engine.unload()


# ---------------------------------------------------------------------------
# Mock generation — FLUX
# ---------------------------------------------------------------------------


class TestFluxGeneration:
    @patch("musicvision.imaging.flux_engine.clear_vram")
    def test_load_and_generate(self, mock_clear, tmp_path, gpu_device_map):
        engine = FluxEngine(ImageGenConfig(), gpu_device_map)
        mock_pipe, mock_image = _mock_pipe()
        engine._pipe = mock_pipe

        output = tmp_path / "test.png"
        result = engine.generate("A cinematic shot", output_path=output)

        assert result.prompt == "A cinematic shot"
        assert result.width == 1280
        assert result.height == 720
        mock_image.save.assert_called_once_with(output)

    @patch("musicvision.imaging.flux_engine.clear_vram")
    def test_schnell_clamps_steps(self, mock_clear, tmp_path, gpu_device_map):
        cfg = ImageGenConfig(model=ImageModel.FLUX_SCHNELL, steps=28)
        engine = FluxEngine(cfg, gpu_device_map)

        mock_pipe, mock_image = _mock_pipe()
        engine._pipe = mock_pipe

        output = tmp_path / "test.png"
        result = engine.generate("test", output_path=output)

        # Schnell should clamp steps to 4 and guidance to 0.0
        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["num_inference_steps"] == 4
        assert call_kwargs["guidance_scale"] == 0.0
        assert result.metadata["steps"] == 4

    @patch("musicvision.imaging.flux_engine.clear_vram")
    def test_lora_load_and_swap(self, mock_clear, tmp_path, gpu_device_map):
        engine = FluxEngine(ImageGenConfig(), gpu_device_map)
        mock_pipe, mock_image = _mock_pipe()
        engine._pipe = mock_pipe

        # First generation with LoRA A
        engine.generate("test", lora_path="/loras/a.safetensors", output_path=tmp_path / "a.png")
        mock_pipe.load_lora_weights.assert_called_once_with("/loras/a.safetensors")

        # Same LoRA — no reload
        mock_pipe.load_lora_weights.reset_mock()
        engine.generate("test", lora_path="/loras/a.safetensors", output_path=tmp_path / "a2.png")
        mock_pipe.load_lora_weights.assert_not_called()

        # Swap to LoRA B
        engine.generate("test", lora_path="/loras/b.safetensors", output_path=tmp_path / "b.png")
        mock_pipe.unload_lora_weights.assert_called_once()
        mock_pipe.load_lora_weights.assert_called_once_with("/loras/b.safetensors")

    @patch("musicvision.imaging.flux_engine.clear_vram")
    def test_unload_cleans_lora(self, mock_clear, gpu_device_map):
        engine = FluxEngine(ImageGenConfig(), gpu_device_map)
        mock_pipe, _ = _mock_pipe()
        engine._pipe = mock_pipe
        engine._current_lora = "/loras/test.safetensors"

        engine.unload()
        mock_pipe.unload_lora_weights.assert_called_once()
        assert not engine.is_loaded
        assert engine._current_lora is None
        mock_clear.assert_called_once()


# ---------------------------------------------------------------------------
# Mock generation — Z-Image
# ---------------------------------------------------------------------------


class TestZImageGeneration:
    @patch("musicvision.imaging.zimage_engine.clear_vram")
    def test_generate(self, mock_clear, tmp_path, gpu_device_map):
        cfg = ImageGenConfig(model=ImageModel.ZIMAGE, steps=20, guidance_scale=3.5)
        engine = ZImageEngine(cfg, gpu_device_map)
        mock_pipe, mock_image = _mock_pipe()
        engine._pipe = mock_pipe

        output = tmp_path / "test.png"
        result = engine.generate("A landscape", output_path=output)

        assert result.prompt == "A landscape"
        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["num_inference_steps"] == 20
        assert call_kwargs["guidance_scale"] == 3.5
        mock_image.save.assert_called_once_with(output)

    @patch("musicvision.imaging.zimage_engine.clear_vram")
    def test_turbo_clamps_steps_and_guidance(self, mock_clear, tmp_path, gpu_device_map):
        cfg = ImageGenConfig(model=ImageModel.ZIMAGE_TURBO, steps=28, guidance_scale=3.5)
        engine = ZImageEngine(cfg, gpu_device_map)
        mock_pipe, mock_image = _mock_pipe()
        engine._pipe = mock_pipe

        output = tmp_path / "test.png"
        result = engine.generate("test", output_path=output)

        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["num_inference_steps"] == 8
        assert call_kwargs["guidance_scale"] == 1.0  # clamped from 3.5
        assert result.metadata["steps"] == 8


# ---------------------------------------------------------------------------
# ImageResult
# ---------------------------------------------------------------------------


class TestImageResult:
    def test_basic_fields(self):
        result = ImageResult(
            path=Path("/images/test.png"),
            seed=42,
            prompt="test prompt",
            width=1280,
            height=720,
        )
        assert result.path == Path("/images/test.png")
        assert result.seed == 42
        assert result.metadata == {}

    def test_metadata(self):
        result = ImageResult(
            path=Path("/images/test.png"),
            seed=42,
            prompt="test",
            width=1280,
            height=720,
            metadata={"steps": 28, "guidance_scale": 3.5},
        )
        assert result.metadata["steps"] == 28
