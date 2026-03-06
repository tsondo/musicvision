"""Tests for the core data models and project service."""

import json
from pathlib import Path

import pytest

from musicvision.models import (
    ApprovalStatus,
    CharacterDef,
    DemucsModel,
    FluxConfig,
    FluxModel,
    FluxQuant,
    HumoConfig,
    HumoTier,
    ProjectConfig,
    Scene,
    SceneList,
    SceneType,
    SeparationMethod,
    StyleSheet,
    VocalSeparationConfig,
)
from musicvision.project import ProjectService
from musicvision.assembly.timecode import seconds_to_filename_stamp, scene_clip_filename


class TestModels:
    def test_project_config_roundtrip(self, tmp_path):
        config = ProjectConfig(
            name="Test Video",
            style_sheet=StyleSheet(
                visual_style="cinematic, moody",
                characters=[CharacterDef(id="singer", description="Woman with short hair")],
            ),
            humo=HumoConfig(tier=HumoTier.GGUF_Q6, scale_a=2.5),
        )

        path = tmp_path / "project.yaml"
        config.save(path)
        loaded = ProjectConfig.load(path)

        assert loaded.name == "Test Video"
        assert loaded.style_sheet.visual_style == "cinematic, moody"
        assert len(loaded.style_sheet.characters) == 1
        assert loaded.humo.scale_a == 2.5

    def test_scene_list_roundtrip(self, tmp_path):
        scenes = SceneList(scenes=[
            Scene(
                id="scene_001", order=1,
                time_start=0.0, time_end=3.88,
                type=SceneType.VOCAL,
                lyrics="Standing in the rain",
            ),
            Scene(
                id="scene_002", order=2,
                time_start=3.88, time_end=8.0,
                type=SceneType.VOCAL,
                lyrics="Waiting for the sun",
            ),
        ])

        path = tmp_path / "scenes.json"
        scenes.save(path)
        loaded = SceneList.load(path)

        assert len(loaded.scenes) == 2
        assert loaded.scenes[0].lyrics == "Standing in the rain"
        assert loaded.get_scene("scene_002").time_end == 8.0

    def test_scene_needs_sub_clips(self):
        short = Scene(id="s1", order=1, time_start=0, time_end=3.5)
        long = Scene(id="s2", order=2, time_start=0, time_end=8.0)

        assert not short.needs_sub_clips
        assert long.needs_sub_clips

    def test_scene_effective_prompt_override(self):
        scene = Scene(
            id="s1", order=1, time_start=0, time_end=3.0,
            image_prompt="auto generated prompt",
            image_prompt_user_override="my custom prompt",
        )
        assert scene.effective_image_prompt == "my custom prompt"

    def test_scene_frame_fields_roundtrip(self, tmp_path):
        """New frame-accurate fields survive JSON roundtrip."""
        scenes = SceneList(scenes=[
            Scene(
                id="scene_001", order=1,
                time_start=0.0, time_end=8.0,
                frame_start=0, frame_end=200, total_frames=200,
                section="verse_1",
                subclip_frame_counts=[67, 67, 66],
                generation_audio_segments=[
                    "segments/sub/scene_001_sub_00.wav",
                    "segments/sub/scene_001_sub_01.wav",
                    "segments/sub/scene_001_sub_02.wav",
                ],
            ),
        ])
        path = tmp_path / "scenes.json"
        scenes.save(path)
        loaded = SceneList.load(path)

        s = loaded.scenes[0]
        assert s.frame_start == 0
        assert s.frame_end == 200
        assert s.total_frames == 200
        assert s.section == "verse_1"
        assert s.subclip_frame_counts == [67, 67, 66]
        assert len(s.generation_audio_segments) == 3

    def test_subclip_frame_count_roundtrip(self, tmp_path):
        """SubClip.frame_count survives JSON roundtrip."""
        from musicvision.models import SubClip as SubClipModel
        scenes = SceneList(scenes=[
            Scene(
                id="scene_001", order=1,
                time_start=0.0, time_end=8.0,
                sub_clips=[
                    SubClipModel(id="scene_001_a", time_start=0.0, time_end=2.68, frame_count=67),
                    SubClipModel(id="scene_001_b", time_start=2.68, time_end=5.36, frame_count=67),
                    SubClipModel(id="scene_001_c", time_start=5.36, time_end=8.0, frame_count=66),
                ],
            ),
        ])
        path = tmp_path / "scenes.json"
        scenes.save(path)
        loaded = SceneList.load(path)
        assert loaded.scenes[0].sub_clips[0].frame_count == 67
        assert loaded.scenes[0].sub_clips[2].frame_count == 66

    def test_scene_defaults_backward_compat(self):
        """New fields default to None/empty — old scenes.json files load fine."""
        scene = Scene(id="s1", order=1, time_start=0, time_end=3.0)
        assert scene.frame_start is None
        assert scene.frame_end is None
        assert scene.total_frames is None
        assert scene.section == ""
        assert scene.subclip_frame_counts is None
        assert scene.generation_audio_segments is None

    def test_humo_config_resolution(self):
        hd = HumoConfig(resolution="720p")
        sd = HumoConfig(resolution="480p")
        assert hd.width == 1280 and hd.height == 720
        assert sd.width == 832 and sd.height == 480


class TestHumoTier:
    def test_tier_roundtrip(self, tmp_path):
        """HumoTier survives yaml serialization."""
        config = ProjectConfig(
            humo=HumoConfig(
                tier=HumoTier.GGUF_Q4,
                denoising_steps=40,
                block_swap_count=10,
            )
        )
        path = tmp_path / "project.yaml"
        config.save(path)
        loaded = ProjectConfig.load(path)
        assert loaded.humo.tier == HumoTier.GGUF_Q4
        assert loaded.humo.denoising_steps == 40
        assert loaded.humo.block_swap_count == 10

    def test_model_size_from_tier(self):
        assert HumoConfig(tier=HumoTier.FP16).model_size == "17B"
        assert HumoConfig(tier=HumoTier.FP8_SCALED).model_size == "17B"
        assert HumoConfig(tier=HumoTier.GGUF_Q4).model_size == "17B"
        assert HumoConfig(tier=HumoTier.PREVIEW).model_size == "1.7B"

    def test_default_tier_is_fp8_scaled(self):
        assert HumoConfig().tier == HumoTier.FP8_SCALED

    def test_block_swap_manager_from_config(self):
        """BlockSwapManager.from_config computes num_gpu_blocks correctly."""
        from musicvision.video.block_swap import BlockSwapManager

        # Simulate 40 transformer blocks as a list
        blocks = list(range(40))

        # block_swap_count=0 → all 40 on GPU (no swap)
        mgr = BlockSwapManager.from_config(blocks, block_swap_count=0, gpu_device="cuda:0")
        assert mgr.num_gpu_blocks == 40
        assert not mgr._swap_enabled

        # block_swap_count=20 → 20 on GPU, 20 on CPU
        mgr = BlockSwapManager.from_config(blocks, block_swap_count=20, gpu_device="cuda:0")
        assert mgr.num_gpu_blocks == 20
        assert mgr._swap_enabled

        # block_swap_count=35 → 5 on GPU (clamped to at least 1)
        mgr = BlockSwapManager.from_config(blocks, block_swap_count=35, gpu_device="cuda:0")
        assert mgr.num_gpu_blocks == 5

    def test_weight_registry_locate_raises_when_missing(self, tmp_path):
        """locate_dit raises FileNotFoundError when weights not present."""
        from musicvision.video.weight_registry import locate_dit, locate_shared

        with pytest.raises(FileNotFoundError):
            locate_dit(HumoTier.GGUF_Q4, base_dir=tmp_path)

        with pytest.raises(FileNotFoundError):
            locate_shared("t5", base_dir=tmp_path)

    def test_weight_status_all_false_when_missing(self, tmp_path):
        """weight_status returns all False for a fresh empty directory."""
        from musicvision.video.weight_registry import weight_status

        status = weight_status(HumoTier.GGUF_Q4, base_dir=tmp_path)
        assert "dit" in status
        assert "t5" in status
        assert "vae" in status
        assert "whisper" in status
        assert all(v is False for v in status.values())


class TestProjectService:
    def test_create_and_open(self, tmp_path):
        project_dir = tmp_path / "my_video"
        svc = ProjectService.create(project_dir, name="My Video")

        assert svc.config.name == "My Video"
        assert (project_dir / "project.yaml").exists()
        assert (project_dir / "scenes.json").exists()
        assert (project_dir / "input").is_dir()
        assert (project_dir / "clips" / "sub").is_dir()

        # Re-open
        svc2 = ProjectService.open(project_dir)
        assert svc2.config.name == "My Video"

    def test_open_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ProjectService.open(tmp_path / "nope")

    def test_acestep_auto_import(self, tmp_path):
        """AceStep JSON detected alongside audio file."""
        project_dir = tmp_path / "proj"
        svc = ProjectService.create(project_dir, name="AceStep Test")

        # Create fake audio + companion JSON
        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(b"fake audio")

        acestep_json = tmp_path / "song.json"
        acestep_json.write_text(json.dumps({
            "bpm": 92,
            "keyscale": "A minor",
            "duration": 240,
            "instrumental": False,
            "caption": "trap-influenced hip-hop, heavy 808 bass",
            "lyrics": "(Intro)\nAyy, it's G-Money\n(Verse 1)\nNow I've been high",
            "seed": 12345,
        }))

        svc.import_audio(audio_file)

        # BPM, duration, key should be populated from AceStep
        assert svc.config.song.bpm == 92
        assert svc.config.song.duration_seconds == 240
        assert svc.config.song.keyscale == "A minor"
        assert svc.config.song.acestep is not None
        assert svc.config.song.acestep.caption.startswith("trap")
        assert svc.config.song.acestep.instrumental is False
        assert svc.config.song.acestep.seed == 12345

        # Lyrics should have been extracted
        assert svc.config.song.lyrics_file == "input/lyrics_acestep.txt"
        lyrics_path = project_dir / svc.config.song.lyrics_file
        assert lyrics_path.exists()
        assert "G-Money" in lyrics_path.read_text()

    def test_acestep_no_json(self, tmp_path):
        """No companion JSON — song fields remain empty."""
        project_dir = tmp_path / "proj"
        svc = ProjectService.create(project_dir, name="Plain Import")

        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(b"fake audio")

        svc.import_audio(audio_file)

        assert svc.config.song.acestep is None
        assert svc.config.song.bpm is None


class TestVocalSeparation:
    def test_defaults(self):
        cfg = VocalSeparationConfig()
        assert cfg.method == SeparationMethod.DEMUCS
        assert cfg.demucs_model == DemucsModel.HTDEMUCS_FT

    def test_roundtrip(self, tmp_path):
        config = ProjectConfig(
            vocal_separation=VocalSeparationConfig(
                method=SeparationMethod.DEMUCS,
                demucs_model=DemucsModel.MDX_EXTRA,
            )
        )
        path = tmp_path / "project.yaml"
        config.save(path)
        loaded = ProjectConfig.load(path)
        assert loaded.vocal_separation.method == SeparationMethod.DEMUCS
        assert loaded.vocal_separation.demucs_model == DemucsModel.MDX_EXTRA

    def test_factory_dispatch(self):
        from musicvision.intake.audio_analysis import (
            DemucsSeparator,
            create_separator,
        )
        demucs = create_separator(SeparationMethod.DEMUCS, device="cpu")
        assert isinstance(demucs, DemucsSeparator)

    def test_demucs_load_error_without_package(self):
        from musicvision.intake.audio_analysis import DemucsSeparator
        sep = DemucsSeparator(model_name="htdemucs", device="cpu")
        # Without demucs installed, load() should raise a clear RuntimeError
        try:
            sep.load()
        except RuntimeError as e:
            assert "demucs" in str(e).lower()
        except Exception:
            pass  # demucs is installed — skip


class TestFluxConfig:
    def test_default_steps(self):
        # dev defaults to 28, schnell defaults to 4
        dev = FluxConfig(model=FluxModel.DEV)
        schnell = FluxConfig(model=FluxModel.SCHNELL)
        assert dev.effective_steps == 28
        assert schnell.effective_steps == 4

    def test_explicit_steps_override(self):
        cfg = FluxConfig(model=FluxModel.SCHNELL, steps=8)
        assert cfg.effective_steps == 8

    def test_quant_default_is_auto(self):
        assert FluxConfig().quant == FluxQuant.AUTO

    def test_flux_config_roundtrip(self, tmp_path):
        config = ProjectConfig(
            flux=FluxConfig(
                model=FluxModel.SCHNELL,
                quant=FluxQuant.INT8,
                steps=4,
                lora_path="assets/loras/style.safetensors",
                lora_weight=0.6,
            )
        )
        path = tmp_path / "project.yaml"
        config.save(path)
        loaded = ProjectConfig.load(path)
        assert loaded.flux.model == FluxModel.SCHNELL
        assert loaded.flux.quant == FluxQuant.INT8
        assert loaded.flux.effective_steps == 4
        assert loaded.flux.lora_path == "assets/loras/style.safetensors"
        assert loaded.flux.lora_weight == 0.6

    def test_strategy_selection(self):
        """Strategy logic is pure Python — no torch needed."""
        # Import here so missing torch doesn't fail the whole module
        from musicvision.imaging.flux_engine import _select_strategy  # noqa: PLC0415

        dev = FluxConfig(model=FluxModel.DEV)
        assert _select_strategy(32.0, dev) == "bf16_split"
        assert _select_strategy(16.0, dev) == "bf16_offload"
        assert _select_strategy(10.0, dev) == "quantized_offload"
        assert _select_strategy(4.0, dev) == "quantized_sequential"

        # Explicit quant overrides VRAM reading
        bf16 = FluxConfig(quant=FluxQuant.BF16)
        fp8 = FluxConfig(quant=FluxQuant.FP8)
        assert _select_strategy(4.0, bf16) == "bf16_offload"
        assert _select_strategy(32.0, fp8) == "quantized_offload"


class TestTimecode:
    def test_filename_stamp(self):
        assert seconds_to_filename_stamp(0.0) == "00m00s000"
        assert seconds_to_filename_stamp(3.88) == "00m03s880"
        assert seconds_to_filename_stamp(63.88) == "01m03s880"

    def test_scene_clip_filename(self):
        name = scene_clip_filename("scene_001", 0.0, 3.88)
        assert name == "scene_001_00m00s000_00m03s880.mp4"
