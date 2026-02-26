#!/usr/bin/env python3
"""
GPU integration test: full HuMo TIA pipeline on real hardware.

Tests (in order):
  1. Hardware detection + weight availability
  2. Smoke-test HuMo inference — single ~4s clip from test assets
  3. generate_scene() — scene > 3.88s, exercises sub-clip splitting + continuity
  4. assemble_rough_cut() — concat sub-clips, mux original audio, verify timing

Usage
-----
  # Fully self-contained (synthesizes test audio + image):
  python scripts/test_gpu_pipeline.py

  # Use real assets:
  python scripts/test_gpu_pipeline.py --audio /path/to/song.wav --image /path/to/ref.png

  # Override tier and step count (faster for iteration):
  python scripts/test_gpu_pipeline.py --tier gguf_q4 --steps 20

  # Use block swap to reduce VRAM (N blocks on CPU):
  python scripts/test_gpu_pipeline.py --block-swap 20

Options
-------
  --tier        HumoTier: fp16 / fp8_scaled / gguf_q6 / gguf_q4 / preview
                Default: auto-detected from VRAM
  --audio       Path to a real WAV/MP3 (8–12s recommended).
                Default: synthesized 8s test tone
  --image       Path to a real reference PNG/JPG.
                Default: generated gradient placeholder
  --steps       Denoising steps (default 15)
  --block-swap  Number of DiT blocks to keep on CPU (default 0 = all on GPU)
  --out-dir     Output directory (default: ./test_output/<timestamp>)
  --preview     Quick smoke test: preview tier + 480p + 10 steps
  --draft       Iteration mode: fp8_scaled tier + 480p + 15 steps
  --skip-encode Skip audio/image encoding stubs if weights are absent
                (generates a latent from noise only — tests denoising loop shape)
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_gpu_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# ANSI helpers
# ─────────────────────────────────────────────────────────────────────────────

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def ok(msg):   return f"{_GREEN}PASS{_RESET}  {msg}"
def err(msg):  return f"{_RED}FAIL{_RESET}  {msg}"
def warn(msg): return f"{_YELLOW}WARN{_RESET}  {msg}"
def hdr(msg):  return f"\n{_BOLD}{_CYAN}{'─'*60}\n{msg}\n{'─'*60}{_RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic asset generation
# ─────────────────────────────────────────────────────────────────────────────

def make_test_audio(path: Path, duration: float = 8.0, sr: int = 44100) -> Path:
    """
    Synthesize a test WAV: layered 440 Hz + 880 Hz tones with rhythmic
    amplitude modulation at 120 BPM (0.5s beats).  Sounds like a simple
    electronic pulse — realistic enough to exercise Whisper feature extraction.
    """
    import numpy as np
    try:
        import soundfile as sf
    except ImportError:
        raise RuntimeError("soundfile not installed: pip install soundfile")

    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)

    # Carrier tones
    sig = 0.4 * np.sin(2 * np.pi * 440 * t)
    sig += 0.2 * np.sin(2 * np.pi * 880 * t)
    sig += 0.1 * np.sin(2 * np.pi * 220 * t)

    # Rhythmic envelope at 120 BPM (beat every 0.5s)
    beat_env = 0.5 * (1 + np.sin(2 * np.pi * 2.0 * t)) ** 2
    sig *= beat_env

    # Slight fade in/out
    fade = int(sr * 0.05)
    sig[:fade]  *= np.linspace(0, 1, fade)
    sig[-fade:] *= np.linspace(1, 0, fade)

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), sig, sr)
    log.info("Synthesized test audio: %s (%.1fs @ %d Hz)", path.name, duration, sr)
    return path


def make_test_image(path: Path, width: int = 1280, height: int = 720) -> Path:
    """
    Generate a reference image: diagonal gradient in purple tones with
    a centered white circle (simulates a performer silhouette).
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
    except ImportError:
        raise RuntimeError("Pillow not installed: pip install Pillow")

    # Gradient background
    xs = np.linspace(0, 1, width, dtype=np.float32)
    ys = np.linspace(0, 1, height, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    diag = (xv + yv) / 2.0

    r = (diag * 80 + 40).astype(np.uint8)
    g = (diag * 20 + 10).astype(np.uint8)
    b = (diag * 180 + 60).astype(np.uint8)
    img_np = np.stack([r, g, b], axis=2)
    img = Image.fromarray(img_np, mode="RGB")

    # Centered white ellipse (performer placeholder)
    draw = ImageDraw.Draw(img)
    cx, cy = width // 2, height // 2
    rx, ry = width // 6, height // 3
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(240, 230, 255))

    # Label
    try:
        draw.text((20, 20), "MusicVision test reference", fill=(255, 255, 255))
    except Exception:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
    log.info("Generated test image: %s (%dx%d)", path.name, width, height)
    return path


def slice_audio_segment(src: Path, dst: Path, start: float, end: float) -> Path:
    """Slice a segment from src WAV using ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.6f}",
        "-i", str(src),
        "-t", f"{end - start:.6f}",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg slice failed: {r.stderr.decode()}")
    return dst


def probe_video(path: Path) -> dict:
    """
    Use ffprobe to extract video metadata.
    Returns dict with 'duration', 'width', 'height', 'fps', 'has_video'.
    """
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        import json
        data = json.loads(r.stdout)
        info = {"has_video": False, "duration": 0.0, "width": 0, "height": 0, "fps": 0.0}
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                info["has_video"] = True
                info["width"] = stream.get("width", 0)
                info["height"] = stream.get("height", 0)
                fps_str = stream.get("r_frame_rate", "0/1")
                try:
                    num, den = fps_str.split("/")
                    info["fps"] = round(float(num) / float(den), 2)
                except Exception:
                    pass
        info["duration"] = float(data.get("format", {}).get("duration", 0))
        return info
    except Exception as e:
        return {"has_video": False, "error": str(e), "duration": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# VRAM snapshot
# ─────────────────────────────────────────────────────────────────────────────

def vram_snapshot() -> str:
    """Return a one-line VRAM summary for all GPUs."""
    try:
        import torch
        parts = []
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            parts.append(f"GPU{i} {alloc:.1f}/{total:.1f} GB")
        return "  ".join(parts) if parts else "no CUDA"
    except Exception:
        return "no CUDA"


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

class Results:
    def __init__(self):
        self.rows: list[tuple[str, bool, float, str]] = []  # (name, passed, elapsed, note)

    def record(self, name: str, passed: bool, elapsed: float, note: str = ""):
        self.rows.append((name, passed, elapsed, note))
        symbol = ok(name) if passed else err(name)
        timing = f"{elapsed:.1f}s"
        print(f"  {symbol}  [{timing}]" + (f"  {note}" if note else ""))

    def summary(self):
        total  = len(self.rows)
        passed = sum(1 for _, p, _, _ in self.rows if p)
        print(hdr(f"Results: {passed}/{total} passed"))
        for name, p, elapsed, note in self.rows:
            sym = f"{_GREEN}✓{_RESET}" if p else f"{_RED}✗{_RESET}"
            print(f"  {sym}  {name:<45} {elapsed:>6.1f}s   {note}")
        return passed == total


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Hardware detection
# ─────────────────────────────────────────────────────────────────────────────

def phase_hardware(args, results: Results) -> tuple:
    """
    Detect GPUs, resolve tier, check weights.
    Returns (device_map, humo_config, tier_name) or raises.
    """
    print(hdr("Phase 1 — Hardware detection"))
    t0 = time.time()

    import torch
    from musicvision.utils.gpu import detect_devices, vram_info
    from musicvision.models import HumoTier, HumoConfig
    from musicvision.video.weight_registry import weight_status

    device_map = detect_devices()
    vinfo = vram_info()
    for gpu in vinfo:
        log.info(
            "  GPU%d  %s  %.1f GB free / %.1f GB total  (CC %s)",
            gpu["index"], gpu["name"], gpu["free_gb"], gpu["total_gb"],
            gpu["compute_capability"],
        )

    # Tier selection
    if args.tier:
        tier = HumoTier(args.tier)
        log.info("Tier override: %s", tier.value)
    else:
        from musicvision.utils.gpu import recommend_tier
        tier = recommend_tier(device_map)
        log.info("Auto-selected tier: %s", tier.value)

    if getattr(args, "fast", False):
        from musicvision.models import HumoQuality
        humo_cfg = HumoConfig.from_quality(
            HumoQuality.FAST,
            block_swap_count=args.block_swap,
            denoising_steps=args.steps,
        )
    else:
        humo_cfg = HumoConfig(
            tier=tier,
            resolution=args.resolution,
            denoising_steps=args.steps,
            block_swap_count=args.block_swap,
            scale_a=2.0,
            scale_t=7.5,
        )

    results.record("GPU detected", torch.cuda.is_available(), time.time() - t0,
                   f"tier={tier.value}, {vram_snapshot()}")

    # Weight check
    t1 = time.time()
    status = weight_status(tier)
    all_present = all(status.values())
    missing = [k for k, v in status.items() if not v]
    results.record(
        "Weights present",
        all_present,
        time.time() - t1,
        f"missing: {missing}" if missing else f"all OK ({', '.join(status)})",
    )

    if missing:
        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if token:
            log.info("Downloading missing weights: %s", missing)
            from musicvision.video.weight_registry import download_all_for_tier
            download_all_for_tier(tier, hf_token=token)
        else:
            log.warning(
                "Missing weights and no HUGGINGFACE_TOKEN found.\n"
                "Run:  musicvision download-weights --tier %s\n"
                "or set HUGGINGFACE_TOKEN in .env",
                tier.value,
            )

    return device_map, humo_cfg, tier.value


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Smoke-test: single clip (~4s)
# ─────────────────────────────────────────────────────────────────────────────

def phase_single_clip(
    args,
    results: Results,
    device_map,
    humo_cfg,
    audio_path: Path,
    image_path: Path,
    out_dir: Path,
) -> Path | None:
    """
    Load HuMo engine and generate one short clip (~3.88s).
    Returns output clip path on success, None on failure.
    """
    print(hdr("Phase 2 — Single clip smoke test"))

    from musicvision.video.humo_engine import HumoEngine, HumoInput

    # Slice first 3.8s of test audio
    t0 = time.time()
    seg = out_dir / "segments" / "smoke_test.wav"
    seg.parent.mkdir(parents=True, exist_ok=True)
    try:
        slice_audio_segment(audio_path, seg, 0.0, 3.8)
        results.record("Slice audio segment", True, time.time() - t0)
    except Exception as e:
        results.record("Slice audio segment", False, time.time() - t0, str(e))
        return None

    # Load engine
    t1 = time.time()
    engine = HumoEngine(humo_cfg, device_map)
    try:
        engine.load()
        load_elapsed = time.time() - t1
        results.record("Engine load()", True, load_elapsed, vram_snapshot())
    except Exception as e:
        results.record("Engine load()", False, time.time() - t1, str(e))
        log.exception("Engine load failed")
        return None

    # Generate clip
    t2 = time.time()
    clip_out = out_dir / "clips" / "smoke_test.mp4"
    clip_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = engine.generate(HumoInput(
            text_prompt=(
                "A performer on stage, dramatic lighting, music video aesthetic, "
                "cinematic 4K, smooth motion, lip synced to beat"
            ),
            reference_image=image_path,
            audio_segment=seg,
            output_path=clip_out,
            seed=42,
        ))
        gen_elapsed = time.time() - t2

        # Validate the output file
        meta = probe_video(clip_out)
        valid = meta["has_video"] and meta["duration"] > 0.5
        note = (
            f"{meta['duration']:.2f}s  {meta['width']}x{meta['height']}  "
            f"{meta['fps']}fps  {vram_snapshot()}"
        )
        results.record("Generate single clip", valid, gen_elapsed, note)

        if not valid:
            log.error("Output video failed probe: %s", meta)
            return None

        log.info("Single clip OK: %s", clip_out)
        return clip_out

    except Exception as e:
        results.record("Generate single clip", False, time.time() - t2, str(e))
        log.exception("generate() failed")
        return None
    finally:
        engine.unload()
        log.info("Engine unloaded. %s", vram_snapshot())


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — generate_scene(): sub-clip splitting
# ─────────────────────────────────────────────────────────────────────────────

def phase_scene_split(
    args,
    results: Results,
    device_map,
    humo_cfg,
    audio_path: Path,
    image_path: Path,
    out_dir: Path,
) -> list[Path]:
    """
    Test generate_scene() with a 7.5s scene — should split into 2 sub-clips
    and use the last frame of sub-clip 0 as the reference for sub-clip 1.
    """
    print(hdr("Phase 3 — generate_scene() with sub-clip splitting"))

    from musicvision.video.humo_engine import HumoEngine

    SCENE_ID    = "scene_test"
    SCENE_DUR   = 7.5   # forces 2 sub-clips (> 3.88s)
    SUB_DUR     = 3.75  # each sub-clip

    seg_dir = out_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Slice full scene audio + two sub-clip audio files
    t0 = time.time()
    try:
        scene_audio = seg_dir / f"{SCENE_ID}.wav"
        slice_audio_segment(audio_path, scene_audio, 0.0, SCENE_DUR)

        # generate_scene() looks for <scene_id>_sub_00.wav, _sub_01.wav etc.
        # in the same directory as audio_segment
        slice_audio_segment(audio_path, seg_dir / f"{SCENE_ID}_sub_00.wav", 0.0, SUB_DUR)
        slice_audio_segment(audio_path, seg_dir / f"{SCENE_ID}_sub_01.wav", SUB_DUR, SCENE_DUR)
        results.record("Slice scene + sub-clip audio", True, time.time() - t0)
    except Exception as e:
        results.record("Slice scene + sub-clip audio", False, time.time() - t0, str(e))
        return []

    # Load engine (reload fresh — unload happened in phase 2)
    t1 = time.time()
    engine = HumoEngine(humo_cfg, device_map)
    try:
        engine.load()
        results.record("Engine reload()", True, time.time() - t1, vram_snapshot())
    except Exception as e:
        results.record("Engine reload()", False, time.time() - t1, str(e))
        log.exception("Engine reload failed")
        return []

    # Run generate_scene()
    t2 = time.time()
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    try:
        outputs = engine.generate_scene(
            text_prompt=(
                "Vocalist performing under neon lights, close-up and mid-shot alternating, "
                "music video style, cinematic, sync with beat"
            ),
            reference_image=image_path,
            audio_segment=scene_audio,
            output_dir=clips_dir,
            scene_id=SCENE_ID,
            duration=SCENE_DUR,
        )
        gen_elapsed = time.time() - t2

        # Expect exactly 2 sub-clips
        n_clips = len(outputs)
        results.record(
            "generate_scene() returns 2 sub-clips",
            n_clips == 2,
            gen_elapsed,
            f"{n_clips} clip(s) generated",
        )

        # Validate each sub-clip
        clip_paths: list[Path] = []
        for i, out in enumerate(outputs):
            t_v = time.time()
            meta = probe_video(out.video_path)
            valid = meta["has_video"] and meta["duration"] > 0.5
            results.record(
                f"Sub-clip {i} valid",
                valid,
                time.time() - t_v,
                f"{meta.get('duration', 0):.2f}s  {meta.get('width')}x{meta.get('height')}",
            )
            if valid:
                clip_paths.append(out.video_path)

        # Verify sub-clip continuity: sub-clip 1 reference is last frame of sub-clip 0
        t_c = time.time()
        lastframe = clips_dir / f"{SCENE_ID}_a_lastframe.png"
        continuity_ok = lastframe.exists()
        results.record(
            "Sub-clip continuity (last-frame ref exists)",
            continuity_ok,
            time.time() - t_c,
            str(lastframe) if continuity_ok else "file not found",
        )

        return clip_paths

    except Exception as e:
        results.record("generate_scene()", False, time.time() - t2, str(e))
        log.exception("generate_scene() failed")
        return []
    finally:
        engine.unload()
        log.info("Engine unloaded. %s", vram_snapshot())


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — assemble_rough_cut()
# ─────────────────────────────────────────────────────────────────────────────

def phase_assembly(
    args,
    results: Results,
    clip_paths: list[Path],
    audio_path: Path,
    out_dir: Path,
) -> Path | None:
    """
    Assemble all available clips into a rough cut with original audio muxed back.
    Validates duration matches sum of input clips.
    """
    print(hdr("Phase 4 — Assembly (concat + audio mux)"))

    if not clip_paths:
        print(f"  {warn('No clips available — skipping assembly')}")
        return None

    from musicvision.models import Scene, SceneList, ApprovalStatus
    from musicvision.assembly.concatenator import assemble_rough_cut
    from musicvision.utils.paths import ProjectPaths

    # Build a minimal SceneList from the generated clips
    scenes_list: list[Scene] = []
    t_cursor = 0.0
    for i, clip_path in enumerate(clip_paths):
        meta = probe_video(clip_path)
        dur = meta.get("duration", 3.8)
        s = Scene(
            id=f"scene_{i:03d}",
            order=i,
            time_start=t_cursor,
            time_end=t_cursor + dur,
            video_clip=str(clip_path),
            video_status=ApprovalStatus.APPROVED,
        )
        scenes_list.append(s)
        t_cursor += dur

    scene_list = SceneList(scenes=scenes_list)
    paths = ProjectPaths(out_dir)
    paths.scaffold()

    t0 = time.time()
    try:
        rough_cut = assemble_rough_cut(
            scenes=scene_list,
            paths=paths,
            original_audio=audio_path,
            approved_only=True,
        )
        elapsed = time.time() - t0

        meta = probe_video(rough_cut)
        valid = meta["has_video"] and meta["duration"] > 1.0
        expected_dur = t_cursor
        dur_ok = abs(meta["duration"] - expected_dur) < 1.5  # allow 1.5s tolerance

        results.record(
            "assemble_rough_cut()",
            valid,
            elapsed,
            f"{meta['duration']:.2f}s (expected ~{expected_dur:.2f}s)",
        )
        results.record(
            "Rough cut duration within tolerance",
            dur_ok,
            0.0,
            f"|{meta['duration']:.2f} - {expected_dur:.2f}| < 1.5s",
        )

        if valid:
            log.info("Rough cut saved: %s", rough_cut)
            return rough_cut
        return None

    except Exception as e:
        results.record("assemble_rough_cut()", False, time.time() - t0, str(e))
        log.exception("Assembly failed")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPU integration test for HuMo TIA pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tier",       default=None,
                        choices=["fp16","fp8_scaled","gguf_q8","gguf_q6","gguf_q4","preview"],
                        help="HuMo precision tier (default: auto-detected)")
    parser.add_argument("--audio",      type=Path, default=None,
                        help="Path to source WAV/MP3 (default: synthesized)")
    parser.add_argument("--image",      type=Path, default=None,
                        help="Path to reference PNG/JPG (default: generated)")
    parser.add_argument("--steps",      type=int, default=15,
                        help="Denoising steps (default: 15)")
    parser.add_argument("--block-swap", type=int, default=0, dest="block_swap",
                        help="DiT blocks to keep on CPU (default: 0 = all on GPU)")
    parser.add_argument("--out-dir",    type=Path, default=None, dest="out_dir",
                        help="Output directory (default: ./test_output/<timestamp>)")
    parser.add_argument("--resolution", default="480p", choices=["720p", "480p", "384p"],
                        help="Output resolution (default: 480p)")
    parser.add_argument("--preview",    action="store_true",
                        help="Quick smoke test: preview tier + 480p + 10 steps")
    parser.add_argument("--draft",      action="store_true",
                        help="Iteration mode: fp8_scaled tier + 480p + 15 steps")
    parser.add_argument("--fast",       action="store_true",
                        help="Lightx2V LoRA mode: fp8_scaled + 384p + 6 steps + CFG=1")
    parser.add_argument("--phase",      type=int, default=0,
                        help="Run only this phase (1=hw, 2=single, 3=scene, 4=assembly; 0=all)")
    args = parser.parse_args()

    # Convenience presets override individual flags
    if args.preview:
        args.tier = args.tier or "preview"
        args.resolution = "480p"
        args.steps = 10
    elif args.fast:
        args.tier = args.tier or "fp8_scaled"
        args.resolution = "384p"
        args.steps = 6
    elif args.draft:
        args.tier = args.tier or "fp8_scaled"
        args.resolution = "480p"
        args.steps = 15

    # Output directory
    if args.out_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = _REPO_ROOT / "test_output" / ts
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", args.out_dir)

    # Prepare test assets
    assets_dir = args.out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    audio_path = args.audio
    if audio_path is None:
        audio_path = assets_dir / "test_audio.wav"
        if not audio_path.exists():
            make_test_audio(audio_path, duration=10.0)
    else:
        audio_path = audio_path.resolve()
        if not audio_path.exists():
            sys.exit(f"Audio file not found: {audio_path}")

    # Resolution dimensions for test image generation
    _RES_DIMS = {"720p": (1280, 720), "480p": (832, 480), "384p": (688, 384)}
    img_w, img_h = _RES_DIMS.get(args.resolution, (832, 480))

    image_path = args.image
    if image_path is None:
        image_path = assets_dir / "test_ref.png"
        if not image_path.exists():
            make_test_image(image_path, width=img_w, height=img_h)
    else:
        image_path = image_path.resolve()
        if not image_path.exists():
            sys.exit(f"Image file not found: {image_path}")

    log.info("Audio: %s", audio_path)
    log.info("Image: %s", image_path)

    results = Results()
    run_all = args.phase == 0

    # Phase 1
    if run_all or args.phase == 1:
        device_map, humo_cfg, tier_name = phase_hardware(args, results)
    else:
        # Still need these for later phases if jumping in
        from musicvision.utils.gpu import detect_devices
        from musicvision.models import HumoTier, HumoConfig
        device_map = detect_devices()
        if getattr(args, "fast", False):
            from musicvision.models import HumoQuality
            humo_cfg = HumoConfig.from_quality(
                HumoQuality.FAST,
                block_swap_count=args.block_swap,
                denoising_steps=args.steps,
            )
            tier_name = humo_cfg.tier.value
        else:
            tier = HumoTier(args.tier) if args.tier else HumoTier.FP8_SCALED
            humo_cfg = HumoConfig(tier=tier, resolution=args.resolution, denoising_steps=args.steps, block_swap_count=args.block_swap)
            tier_name = tier.value

    clip_paths: list[Path] = []

    # Phase 2
    if run_all or args.phase == 2:
        single_clip = phase_single_clip(
            args, results, device_map, humo_cfg, audio_path, image_path, args.out_dir
        )
        if single_clip:
            clip_paths.append(single_clip)

    # Phase 3
    if run_all or args.phase == 3:
        scene_clips = phase_scene_split(
            args, results, device_map, humo_cfg, audio_path, image_path, args.out_dir
        )
        clip_paths.extend(scene_clips)

    # Phase 4
    if run_all or args.phase == 4:
        rough_cut = phase_assembly(args, results, clip_paths, audio_path, args.out_dir)
        if rough_cut:
            log.info("%s", f"{_BOLD}Final rough cut:{_RESET} {rough_cut}")

    # Summary
    all_passed = results.summary()
    print(f"\nOutput directory: {args.out_dir}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
