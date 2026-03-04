"""
SeedVR2 video upscaling engine (subprocess-based).

ByteDance SeedVR2-3B: one-step diffusion upscaler with good temporal
consistency and face preservation. Runs in a separate venv via JSON IPC
(same pattern as HunyuanVideo-Avatar).

Env vars: SEEDVR2_REPO_DIR, SEEDVR2_VENV_PYTHON
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from musicvision.upscaling.base import UpscaleEngine, UpscaleInput, UpscaleResult
from musicvision.utils.video import get_video_resolution

log = logging.getLogger(__name__)


class SeedVR2Engine(UpscaleEngine):
    """SeedVR2 upscaler via subprocess bridge."""

    def __init__(
        self,
        repo_dir: str = "",
        venv_python: str = "",
        model_id: str = "ByteDance-Seed/SeedVR2-3B",
        use_fp8: bool = True,
    ):
        self._repo_dir = Path(repo_dir or os.environ.get("SEEDVR2_REPO_DIR", "")).expanduser()
        self._venv_python = venv_python or os.environ.get("SEEDVR2_VENV_PYTHON", "")
        if not self._venv_python and self._repo_dir.exists():
            candidate = self._repo_dir / ".venv" / "bin" / "python"
            if candidate.exists():
                self._venv_python = str(candidate)
        self._model_id = model_id
        self._use_fp8 = use_fp8
        self._loaded = False

    def load(self) -> None:
        if not self._repo_dir.exists():
            raise RuntimeError(
                f"SeedVR2 repo not found at {self._repo_dir}. "
                "Set SEEDVR2_REPO_DIR env var or seedvr2_repo_dir in project config."
            )
        if not self._venv_python or not Path(self._venv_python).exists():
            raise RuntimeError(
                f"SeedVR2 venv python not found: {self._venv_python}. "
                "Set SEEDVR2_VENV_PYTHON or create .venv in the repo dir."
            )
        self._loaded = True
        log.info("SeedVR2 engine ready (repo=%s, fp8=%s)", self._repo_dir, self._use_fp8)

    def upscale(self, input: UpscaleInput) -> UpscaleResult:
        if not self._loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

        source_res = get_video_resolution(input.video_path)
        input.output_path.parent.mkdir(parents=True, exist_ok=True)

        wrapper = Path(__file__).parent.parent.parent.parent / "scripts" / "seedvr2_wrapper.py"
        if not wrapper.exists():
            raise RuntimeError(f"SeedVR2 wrapper script not found: {wrapper}")

        request = {
            "video_path": str(input.video_path),
            "output_path": str(input.output_path),
            "target_width": input.target_width,
            "target_height": input.target_height,
            "model_id": self._model_id,
            "use_fp8": self._use_fp8,
        }

        cmd = [self._venv_python, str(wrapper)]
        log.info("Running SeedVR2: %s → %s", input.video_path.name, input.output_path.name)

        env = os.environ.copy()
        env["SEEDVR2_REPO_DIR"] = str(self._repo_dir)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

        result = subprocess.run(
            cmd,
            input=json.dumps(request),
            capture_output=True, text=True,
            cwd=str(self._repo_dir),
            env=env,
            timeout=1800,  # 30 min timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"SeedVR2 failed: {result.stderr.strip()}")

        response = json.loads(result.stdout.strip().split("\n")[-1])
        if response.get("status") != "success":
            raise RuntimeError(f"SeedVR2 error: {response.get('error', 'unknown')}")

        return UpscaleResult(
            video_path=input.output_path,
            source_resolution=source_res,
            output_resolution=(input.target_width, input.target_height),
            metadata={"model": self._model_id, "fp8": self._use_fp8},
        )

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded
