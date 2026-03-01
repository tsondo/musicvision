"""
HunyuanVideo-Avatar engine — subprocess-based I2V generation.

Runs HunyuanVideo-Avatar in a separate venv via ``scripts/hva_wrapper.py``.
No GPU memory is used by this process — the wrapper subprocess owns the full
model lifecycle.

Lifecycle
---------
    engine = HunyuanAvatarEngine(config)
    engine.load()                        # validates paths, no GPU allocation
    results = engine.generate_scene(...)  # subprocess per sub-clip
    engine.unload()                      # no-op

The wrapper protocol is JSON-based:
  - Write request.json with image/audio/prompt/output paths + gen params
  - Call: <hva_venv_python> scripts/hva_wrapper.py --request req.json --response resp.json
  - Read response.json for status, video_path, error
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import tempfile
from pathlib import Path

from musicvision.models import HunyuanAvatarConfig
from musicvision.video.base import VideoEngine, VideoInput, VideoResult

log = logging.getLogger(__name__)

# Path to the wrapper script relative to the musicvision repo root
_WRAPPER_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "hva_wrapper.py"


def _sub_clip_suffixes(n: int) -> list[str]:
    """Generate sub-clip suffixes: a, b, ..., z, aa, ab, ..."""
    result: list[str] = []
    for i in range(n):
        if i < 26:
            result.append(chr(ord("a") + i))
        else:
            result.append(chr(ord("a") + (i // 26) - 1) + chr(ord("a") + (i % 26)))
    return result


class HunyuanAvatarEngine(VideoEngine):
    """Video engine that delegates to HunyuanVideo-Avatar via subprocess."""

    def __init__(self, config: HunyuanAvatarConfig) -> None:
        self.config = config
        self._loaded = False

    def load(self) -> None:
        """Validate that the HVA repo and venv exist. No GPU memory used.

        Paths resolve in order: project config → env var → error.
        """
        import os

        repo_dir = self.config.hva_repo_dir or os.environ.get("HVA_REPO_DIR", "")
        venv_py = self.config.hva_venv_python or os.environ.get("HVA_VENV_PYTHON", "")

        # Auto-derive venv python from repo dir if not set
        if repo_dir and not venv_py:
            candidate = Path(repo_dir) / ".venv" / "bin" / "python"
            if candidate.is_file():
                venv_py = str(candidate)

        repo = Path(repo_dir)
        if not repo.is_dir():
            raise FileNotFoundError(
                f"HVA repo dir not found: {repo}\n"
                f"Set hva_repo_dir in project.yaml or HVA_REPO_DIR env var."
            )

        venv_python = Path(venv_py)
        if not venv_python.is_file():
            raise FileNotFoundError(
                f"HVA venv python not found: {venv_python}\n"
                f"Set hva_venv_python in project.yaml or HVA_VENV_PYTHON env var."
            )

        # Store resolved paths for use during generation
        self.config.hva_repo_dir = str(repo)
        self.config.hva_venv_python = str(venv_python)

        if not _WRAPPER_SCRIPT.is_file():
            raise FileNotFoundError(f"HVA wrapper script not found: {_WRAPPER_SCRIPT}")

        log.info("HunyuanAvatarEngine validated: repo=%s, python=%s", repo, venv_python)
        self._loaded = True

    def unload(self) -> None:
        """No-op — subprocess owns GPU lifecycle."""
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, input: VideoInput) -> VideoResult:
        """Generate a single video clip via the HVA subprocess wrapper."""
        if not self._loaded:
            raise RuntimeError("Call load() before generate()")

        request = {
            "image_path": str(input.reference_image),
            "audio_path": str(input.audio_segment),
            "prompt": input.text_prompt,
            "output_path": str(input.output_path),
            "hva_repo_dir": self.config.hva_repo_dir,
            "checkpoint": self.config.checkpoint,
            "image_size": self.config.image_size,
            "sample_n_frames": self.config.sample_n_frames,
            "cfg_scale": self.config.cfg_scale,
            "infer_steps": self.config.infer_steps,
            "flow_shift": self.config.flow_shift,
            "seed": self.config.seed,
            "use_deepcache": self.config.use_deepcache,
            "use_fp8": self.config.use_fp8,
            "cpu_offload": self.config.cpu_offload,
        }

        with tempfile.TemporaryDirectory(prefix="hva_req_") as tmpdir:
            req_path = Path(tmpdir) / "request.json"
            resp_path = Path(tmpdir) / "response.json"

            with open(req_path, "w") as f:
                json.dump(request, f, indent=2)

            cmd = [
                self.config.hva_venv_python,
                str(_WRAPPER_SCRIPT),
                "--request", str(req_path),
                "--response", str(resp_path),
            ]

            log.info("Running HVA wrapper: %s → %s", input.reference_image.name, input.output_path.name)

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=660,  # slightly more than wrapper's internal timeout
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"HVA subprocess timed out: {exc}") from exc

            if not resp_path.exists():
                raise RuntimeError(
                    f"HVA wrapper did not produce response. "
                    f"returncode={result.returncode}, stderr={result.stderr[-2000:]}"
                )

            with open(resp_path) as f:
                response = json.load(f)

        if response["status"] != "success":
            raise RuntimeError(f"HVA generation failed: {response['error']}")

        return VideoResult(
            video_path=Path(response["video_path"]),
            frames_generated=response["frames_generated"],
            duration_seconds=response["duration"],
            metadata={"engine": "hunyuan_avatar"},
        )

    def generate_scene(
        self,
        text_prompt: str,
        reference_image: Path,
        audio_segment: Path,
        output_dir: Path,
        scene_id: str,
        duration: float,
    ) -> list[VideoResult]:
        """
        Generate video for a full scene, splitting into sub-clips if needed.

        HunyuanVideo-Avatar supports longer clips than HuMo (5.16s default vs 3.88s),
        but scenes longer than max_duration still need splitting.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before generate_scene()")

        output_dir.mkdir(parents=True, exist_ok=True)
        max_dur = self.config.max_duration

        if duration <= max_dur:
            output_path = output_dir / f"{scene_id}.mp4"
            result = self.generate(VideoInput(
                text_prompt=text_prompt,
                reference_image=reference_image,
                audio_segment=audio_segment,
                output_path=output_path,
            ))
            return [result]

        # Split into sub-clips
        n_sub = math.ceil(duration / max_dur)
        suffixes = _sub_clip_suffixes(n_sub)
        outputs: list[VideoResult] = []

        # Look for pre-sliced audio segments
        seg_dir = audio_segment.parent

        for i, suffix in enumerate(suffixes):
            sub_audio = seg_dir / f"{scene_id}_sub_{i:02d}.wav"
            if not sub_audio.exists():
                sub_audio = seg_dir / f"{scene_id}_sub_{i:02d}.flac"
            if not sub_audio.exists():
                log.warning("Sub-clip audio not found: %s, using full segment", sub_audio)
                sub_audio = audio_segment

            output_path = output_dir / f"{scene_id}_{suffix}.mp4"

            log.info(
                "Generating sub-clip %s/%s: %s (%.2fs)",
                i + 1, n_sub, output_path.name, min(max_dur, duration - i * max_dur),
            )

            result = self.generate(VideoInput(
                text_prompt=text_prompt,
                reference_image=reference_image,
                audio_segment=sub_audio,
                output_path=output_path,
            ))
            outputs.append(result)

        return outputs
