"""
HunyuanVideo-Avatar engine — subprocess-based I2V generation.

Runs HunyuanVideo-Avatar in a separate venv.  Two modes:

1. **Server mode** (default): A persistent ``scripts/hva_server.py`` subprocess
   loads models once and processes clips via stdin/stdout JSON-line IPC.
   ~3 min/clip instead of ~10 min (avoids ~7 min model reload per clip).

2. **Wrapper mode** (fallback): Spawns ``scripts/hva_wrapper.py`` per clip.
   Used automatically if the server fails to start or dies mid-generation.

Lifecycle
---------
    engine = HunyuanAvatarEngine(config)
    engine.load()                        # start server subprocess
    results = engine.generate_scene(...)  # IPC per sub-clip
    engine.unload()                      # shutdown server, free GPU
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import tempfile
import threading
from pathlib import Path

from musicvision.models import HunyuanAvatarConfig
from musicvision.video.base import VideoEngine, VideoInput, VideoResult

log = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
_WRAPPER_SCRIPT = _SCRIPTS_DIR / "hva_wrapper.py"
_SERVER_SCRIPT = _SCRIPTS_DIR / "hva_server.py"

# Timeouts (seconds)
_SERVER_STARTUP_TIMEOUT = 900   # 15 min — covers full model loading
_SERVER_CLIP_TIMEOUT = 900      # 15 min per clip (long audio segments slow whisper/feature extraction)
_WRAPPER_TIMEOUT = 1800         # 30 min per clip (includes model loading)


def _sub_clip_suffixes(n: int) -> list[str]:
    """Generate sub-clip suffixes: a, b, ..., z, aa, ab, ..."""
    result: list[str] = []
    for i in range(n):
        if i < 26:
            result.append(chr(ord("a") + i))
        else:
            result.append(chr(ord("a") + (i // 26) - 1) + chr(ord("a") + (i % 26)))
    return result


def _drain_stderr(proc: subprocess.Popen) -> None:
    """Read server stderr in a background thread so it doesn't block."""
    try:
        for line in proc.stderr:
            log.info("[hva_server] %s", line.rstrip())
    except (ValueError, OSError):
        pass  # pipe closed


class HunyuanAvatarEngine(VideoEngine):
    """Video engine that delegates to HunyuanVideo-Avatar via subprocess."""

    def __init__(self, config: HunyuanAvatarConfig) -> None:
        self.config = config
        self._loaded = False
        self._server_proc: subprocess.Popen | None = None
        self._server_mode: bool = False
        self._stderr_thread: threading.Thread | None = None

    def load(self) -> None:
        """Validate paths, then start the persistent HVA server.

        Falls back to per-clip wrapper mode if the server fails to start.
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

        # Try to start persistent server
        if _SERVER_SCRIPT.is_file():
            try:
                self._start_server()
                self._server_mode = True
                log.info("HVA persistent server started — models will load once")
            except Exception as exc:
                log.warning("Failed to start HVA server, will use per-clip wrapper mode: %s", exc)
                self._server_mode = False
        else:
            log.info("HVA server script not found, using per-clip wrapper mode")
            self._server_mode = False

        self._loaded = True

    def _start_server(self) -> None:
        """Launch the persistent hva_server.py subprocess and wait for ready."""
        weights_dir = Path(self.config.hva_repo_dir) / "weights"
        ckpt_dir = weights_dir / "ckpts" / "hunyuan-video-t2v-720p" / "transformers"

        if self.config.checkpoint == "fp8":
            ckpt_path = ckpt_dir / "mp_rank_00_model_states_fp8.pt"
        else:
            ckpt_path = ckpt_dir / "mp_rank_00_model_states.pt"

        cmd = [
            self.config.hva_venv_python,
            str(_SERVER_SCRIPT),
            "--hva-repo-dir", self.config.hva_repo_dir,
            "--ckpt", str(ckpt_path),
            "--image-size", str(self.config.image_size),
            "--sample-n-frames", str(self.config.sample_n_frames),
            "--cfg-scale", str(self.config.cfg_scale),
            "--infer-steps", str(self.config.infer_steps),
            "--flow-shift-eval-video", str(self.config.flow_shift),
        ]

        if self.config.seed is not None:
            cmd.extend(["--seed", str(self.config.seed)])
        if self.config.use_deepcache:
            cmd.extend(["--use-deepcache", "1"])
        if self.config.use_fp8:
            cmd.append("--use-fp8")
        if self.config.cpu_offload:
            cmd.append("--cpu-offload")

        log.info("Starting HVA server: %s", " ".join(cmd[:4]) + " ...")

        self._server_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Drain stderr in background so it doesn't block
        self._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(self._server_proc,), daemon=True,
        )
        self._stderr_thread.start()

        # Wait for ready signal
        log.info("Waiting for HVA server to load models (may take ~7 min)...")
        try:
            ready_line = self._read_response(timeout=_SERVER_STARTUP_TIMEOUT)
        except Exception as exc:
            self._kill_server()
            raise RuntimeError(f"HVA server failed during startup: {exc}") from exc

        if ready_line.get("status") == "error":
            self._kill_server()
            raise RuntimeError(f"HVA server startup error: {ready_line.get('error')}")

        if ready_line.get("status") != "ready":
            self._kill_server()
            raise RuntimeError(f"Unexpected server startup response: {ready_line}")

    def _read_response(self, timeout: float = _SERVER_CLIP_TIMEOUT) -> dict:
        """Read a single JSON line from the server's stdout with timeout."""
        import select

        if self._server_proc is None or self._server_proc.stdout is None:
            raise RuntimeError("Server process not running")

        # Use select for timeout on the stdout pipe
        fileno = self._server_proc.stdout.fileno()
        ready, _, _ = select.select([fileno], [], [], timeout)
        if not ready:
            raise TimeoutError(f"HVA server did not respond within {timeout}s")

        line = self._server_proc.stdout.readline()
        if not line:
            # EOF — server died
            rc = self._server_proc.poll()
            raise RuntimeError(f"HVA server process exited unexpectedly (returncode={rc})")

        return json.loads(line)

    def _send_request(self, request: dict) -> None:
        """Write a JSON line to the server's stdin."""
        if self._server_proc is None or self._server_proc.stdin is None:
            raise RuntimeError("Server process not running")

        self._server_proc.stdin.write(json.dumps(request) + "\n")
        self._server_proc.stdin.flush()

    def _kill_server(self) -> None:
        """Force-kill the server process."""
        if self._server_proc is not None:
            try:
                self._server_proc.kill()
                self._server_proc.wait(timeout=10)
            except Exception:
                pass
            self._server_proc = None

    def _server_alive(self) -> bool:
        """Check if the server process is still running."""
        return (
            self._server_proc is not None
            and self._server_proc.poll() is None
        )

    def _drain_stale_response(self) -> None:
        """After a timeout, drain the late response so the next request starts clean."""
        if not self._server_alive():
            return
        import select
        try:
            fileno = self._server_proc.stdout.fileno()
            # Wait up to 10 min for the server to finish the timed-out clip
            ready, _, _ = select.select([fileno], [], [], _SERVER_CLIP_TIMEOUT)
            if ready:
                line = self._server_proc.stdout.readline()
                if line:
                    log.info("Drained stale server response: %.100s...", line.strip())
        except Exception as exc:
            log.debug("Failed to drain stale response: %s", exc)

    def unload(self) -> None:
        """Shut down the persistent server and release GPU memory."""
        if self._server_alive():
            try:
                self._send_request({"command": "shutdown"})
                self._server_proc.wait(timeout=30)
                log.info("HVA server shut down cleanly")
            except Exception as exc:
                log.warning("HVA server did not shut down cleanly: %s", exc)
                self._kill_server()
        self._server_proc = None
        self._server_mode = False
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, input: VideoInput) -> VideoResult:
        """Generate a single video clip, using server or wrapper mode."""
        if not self._loaded:
            raise RuntimeError("Call load() before generate()")

        if self._server_mode and self._server_alive():
            try:
                return self._generate_via_server(input)
            except TimeoutError:
                log.warning(
                    "Server timed out on this clip — falling back to wrapper. "
                    "Server still alive, will retry server for next clip."
                )
                # Drain the stale response that will eventually arrive
                self._drain_stale_response()
                return self._generate_via_wrapper(input)
            except Exception as exc:
                log.warning("Server generation failed, falling back to wrapper for this clip: %s", exc)
                if not self._server_alive():
                    log.warning("Server process died — disabling server mode")
                    self._server_mode = False

        return self._generate_via_wrapper(input)

    def _generate_via_server(self, input: VideoInput) -> VideoResult:
        """Send a clip request to the persistent server and read the response."""
        request = {
            "image_path": str(input.reference_image),
            "audio_path": str(input.audio_segment),
            "prompt": input.text_prompt,
            "output_path": str(input.output_path),
            "sample_n_frames": self.config.sample_n_frames,
            "fps": self.config.fps,
        }

        log.info("Sending to HVA server: %s → %s", input.reference_image.name, input.output_path.name)

        self._send_request(request)
        response = self._read_response(timeout=_SERVER_CLIP_TIMEOUT)

        if response["status"] != "success":
            raise RuntimeError(f"HVA server generation failed: {response.get('error')}")

        return VideoResult(
            video_path=Path(response["video_path"]),
            frames_generated=response["frames_generated"],
            duration_seconds=response["duration"],
            metadata={"engine": "hunyuan_avatar", "mode": "server"},
        )

    def _generate_via_wrapper(self, input: VideoInput) -> VideoResult:
        """Fallback: spawn a fresh hva_wrapper.py subprocess per clip."""
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
                    timeout=_WRAPPER_TIMEOUT,
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
            metadata={"engine": "hunyuan_avatar", "mode": "wrapper"},
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
