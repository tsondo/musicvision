# Cloud GPU Deployment Spec

**Affected files:**
- `src/musicvision/utils/gpu.py` — tier selection fixes for datacenter GPUs
- `src/musicvision/imaging/flux_engine.py` — no-offload path for high-VRAM
- `Dockerfile` — new file
- `docker-compose.yml` — new file
- `scripts/cloud_setup.sh` — new file
- `docs/CLOUD_DEPLOYMENT.md` — new documentation

## Goal

Make MusicVision run effectively on cloud GPU VMs with A100 80GB, H100 80GB, or H200 141GB GPUs. Single-GPU, single-tenant deployment. No multi-tenant orchestration (that's a separate future effort).

---

## Hardware Recommendations

### For personal use / single projects

| GPU | VRAM | Cost (~spot) | Throughput | Recommendation |
|-----|------|-------------|-----------|----------------|
| **A100 80GB** | 80 GB | $1.50-2.50/hr | Baseline | **Best value.** All models run FP16/BF16, no quantization, no splitting. |
| **H100 80GB** | 80 GB | $2.50-4.00/hr | ~1.8x A100 | Worth it if rendering 50+ scenes. FP8 native support gives additional options. |
| **H200 141GB** | 141 GB | $4.00-6.00/hr | ~2x A100 | Overkill for single-user. The VRAM headroom is wasted since models fit in 80GB. |
| **A100 40GB** | 40 GB | $1.00-1.50/hr | Baseline | **Viable but tight.** FP16 FLUX (24GB) + VAE + encoders fit. Video engines need FP8 or quantization. |
| **L40S 48GB** | 48 GB | $1.00-1.80/hr | ~0.8x A100 | Decent mid-tier. All models fit in FP16. Slightly slower than A100 on diffusion workloads. |

**Recommendation: A100 80GB** is the sweet spot. Everything runs at full quality with massive headroom. No model splitting, no offload, no quantization compromises.

### Why single-GPU is better than multi-GPU for cloud

MusicVision's multi-GPU strategy was designed for consumer hardware (32GB + 16GB). On cloud:
- A single A100 80GB has more VRAM than both consumer GPUs combined (48GB)
- No PCIe transfer overhead between GPUs
- No model-splitting complexity
- Simpler VM provisioning (single-GPU instances are cheaper and more available)
- The pipeline runs stages sequentially (FLUX → video engine → upscaler), so a single large GPU is fully utilized

Multi-GPU cloud (e.g., 2×A100) only makes sense for the future multi-job scenario where you want to run two pipeline stages in parallel or serve multiple users.

---

## Code Changes Required

### 1. `gpu.py` — `recommend_tier()` fix

Current issue: the FP16 tier requires `n_gpus >= 2 and primary_gb >= 40`. A single A100 80GB gets `n_gpus == 1, primary_gb == 80`, which falls through to FP8_SCALED.

Fix: the `n_gpus == 1 and primary_gb >= 48` check already exists but only applies to HuMo tiers. Verify it works for all engines. The actual change is small — ensure the single-GPU FP16 threshold covers all engine loading paths.

```python
def recommend_tier(device_map: DeviceMap) -> HumoTier:
    # ... existing MPS check ...

    if n_gpus == 1 and primary_gb >= 48:
        return HumoTier.FP16    # A100 80GB, H100, H200, L40S 48GB
    if n_gpus >= 2 and primary_gb >= 40:
        return HumoTier.FP16    # Consumer dual-GPU (5090 + 4080)
    # ... rest unchanged ...
```

**Test**: Verify on an actual A100 that `detect_devices()` returns a single-GPU DeviceMap and `recommend_tier()` returns FP16.

### 2. `flux_engine.py` — No-offload high-VRAM path

Current issue: `_select_strategy()` may apply CPU offload unnecessarily on 80GB cards. The `bf16_split` strategy already handles single-GPU with ≥28GB free, but verify the logic doesn't accidentally trigger offload.

The fix is in `_select_strategy()`:

```python
def _select_strategy(free_gb: float, config: ImageGenConfig) -> str:
    # If running on a single GPU with lots of VRAM, skip offload entirely
    if free_gb >= 28:
        return "bf16_split"  # misleading name — on single GPU this means "bf16, no offload"
    if free_gb >= 18:
        return "bf16_offload"
    # ... quantization fallbacks ...
```

The `_load_bf16_no_offload()` method already exists and is used when `dit_device == encoder_device` (single GPU). Verify this path is taken on cloud instances.

**Also applies to video engines**: Check that `humo_engine.py` and `ltx_video_engine.py` don't apply unnecessary offload or quantization when running on a single 80GB GPU.

### 3. SeedVR2 upscaler — single-GPU mode

SeedVR2 runs as a subprocess. Verify it works on a single-GPU system where `cuda:0` is the only device.

---

## Dockerfile

```dockerfile
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

# UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# App code
WORKDIR /app
COPY . .

# Python environment
RUN uv sync --extra dev
RUN uv run pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Model weight cache (populated at build time or mounted at runtime)
ENV MUSICVISION_WEIGHTS_DIR=/weights
VOLUME /weights

# Project data (mounted at runtime)
VOLUME /projects

# API server
EXPOSE 8000
CMD ["uv", "run", "musicvision", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

### Weight caching strategy

Model weights (~50GB total) should NOT be baked into the Docker image. Options:

**Option A: Persistent volume (recommended for cloud VMs)**
- Mount a persistent disk at `/weights`
- First run downloads weights; subsequent runs use cache
- Survives VM restarts; no re-download on container restart
- `docker run -v /mnt/weights:/weights -v /mnt/projects:/projects musicvision`

**Option B: Pre-built weight image (recommended for frequent cold starts)**
- Build a separate Docker image with weights baked in
- `FROM musicvision-base AS weights-builder; RUN download_weights.py`
- Larger image (~60GB) but zero cold-start latency
- Use for spot instances that may be reclaimed and restarted frequently

**Option C: Cloud storage mount (S3/GCS FUSE)**
- Mount an S3 or GCS bucket at `/weights` via s3fs-fuse or gcsfuse
- Weights cached on first access; subsequent reads from local cache
- Works across multiple VMs sharing the same bucket
- Higher latency on first load but simplest multi-VM setup

**Recommended**: Option A for single-VM usage, Option C for future multi-VM scaling.

### Weight download script

```bash
#!/bin/bash
# scripts/download_weights.sh
# Downloads all model weights to $MUSICVISION_WEIGHTS_DIR

set -e
DIR="${MUSICVISION_WEIGHTS_DIR:-$HOME/.cache/musicvision/weights}"
mkdir -p "$DIR"

echo "Downloading FLUX schnell..."
huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir "$DIR/flux-schnell"

echo "Downloading Z-Image-Turbo..."
huggingface-cli download Zheng-Peng-Fei/Z-Image-turbo --local-dir "$DIR/z-image-turbo"

echo "Downloading Whisper large-v3..."
huggingface-cli download openai/whisper-large-v3 --local-dir "$DIR/whisper-large-v3"

# HuMo, LTX-2, SeedVR2 weights — add as needed
# These may require HUGGINGFACE_TOKEN for gated repos

echo "Weights downloaded to $DIR"
du -sh "$DIR"
```

---

## Cloud Provider Notes

### RunPod / Vast.ai / Lambda Labs

These are GPU-as-a-service platforms with the simplest setup:
- Provision a single A100 80GB instance
- SSH in, clone repo, run `setup_env.sh`
- Or use the Docker image
- Expose port 8000 for the API, port 5173 for the React frontend (or use SSH tunnel)

**RunPod** has persistent storage volumes that survive instance restarts — ideal for weight caching. They also support custom Docker images.

**Lambda Labs** provides bare-metal Ubuntu with pre-installed CUDA drivers. Simplest path: just clone and run, no Docker needed.

### AWS / GCP / Azure

For cloud-native deployments:
- **AWS**: `p4d.24xlarge` (8×A100 40GB) or `p5.48xlarge` (8×H100 80GB) — overkill for single-user, but these are the instances with A100/H100s. Use only 1 GPU. Or `g5.xlarge` (1×A10G 24GB) for budget preview-quality renders.
- **GCP**: `a2-highgpu-1g` (1×A100 40GB) or `a3-highgpu-1g` (1×H100 80GB)
- **Azure**: `Standard_NC24ads_A100_v4` (1×A100 80GB)

For AWS/GCP/Azure, Docker is the recommended deployment method. Use their container services (ECS, Cloud Run, ACI) or just run Docker on a VM.

### Cost optimization

- **Spot/preemptible instances**: 60-80% cheaper. MusicVision's pipeline is inherently resumable — if an instance is reclaimed mid-render, the generated clips survive on disk. Re-run with `--scene-ids` to pick up where you left off.
- **Auto-shutdown**: Add a cron job or watchdog that shuts down the instance after N minutes of idle (no API requests). Saves money when you're reviewing results locally.
- **Right-size**: Don't rent H100 for a 5-scene test project. Use A100 40GB or even L40S for iteration, upgrade to A100 80GB for final renders.

---

## Environment Variables for Cloud

Add to `.env.example`:

```bash
# Cloud-specific settings
MUSICVISION_WEIGHTS_DIR=/weights          # Override model cache location
MUSICVISION_SINGLE_GPU=1                  # Force single-GPU mode (skip GPU ranking)
MUSICVISION_NO_OFFLOAD=1                  # Disable CPU offload (for 80GB+ GPUs)
MUSICVISION_HOST=0.0.0.0                  # Bind to all interfaces (cloud)
MUSICVISION_PORT=8000                     # API port
```

The `MUSICVISION_SINGLE_GPU` flag is a hint to skip the GPU-ranking logic in `detect_devices()` and just use `cuda:0` for everything. This avoids edge cases with cloud multi-GPU instances where you only want to use one GPU.

---

## Frontend Serving in Cloud

Two options:

**Option A: Separate frontend (recommended for development)**
- Run the React dev server on the cloud VM: `npm run dev -- --host 0.0.0.0`
- Access via SSH tunnel or direct IP
- Hot-reload works for frontend development

**Option B: Static build served by FastAPI (recommended for deployment)**
- Build the React app: `cd frontend && npm run build`
- Serve from FastAPI: mount the `dist/` directory as static files
- Single port, single process, simpler networking

Add to `api/app.py`:

```python
from fastapi.staticfiles import StaticFiles

# Serve React build if it exists
frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
```

---

## Testing Plan

### Local simulation

Before renting cloud time:
1. Run the existing test suite to verify no regressions
2. If you have a single GPU with ≥24GB (the 5090), test single-GPU mode by setting `CUDA_VISIBLE_DEVICES=0` to hide the 4080
3. Verify `detect_devices()` returns a single-GPU DeviceMap
4. Verify `recommend_tier()` returns the expected tier
5. Run a short end-to-end test (3-5 scenes, 320p preview)

### Cloud validation

Rent an A100 80GB spot instance for ~1 hour:
1. Clone repo, run setup
2. Download weights (time this — important for cold-start estimation)
3. Create a test project, run full pipeline
4. Verify: FP16 tier selected, no offload, no quantization warnings
5. Benchmark: time per scene for image gen, video gen, upscale
6. Verify: VRAM usage stays well under 80GB (log output from `log_vram_usage()`)
7. Test the Docker image if built

### Benchmark targets

Expected per-scene times on A100 80GB (rough estimates, verify empirically):

| Stage | A100 80GB | RTX 5090 32GB (current) |
|-------|-----------|------------------------|
| FLUX image gen (1280×720) | ~8s | ~12s |
| HuMo video gen (3.88s clip, 480p) | ~3.5min | ~4.5min |
| SeedVR2 upscale (720p → 1080p) | ~30s | ~40s |
| LTX-2 video gen (10s clip) | ~20s | ~30s |

These are estimates. The A100's advantage is primarily memory bandwidth (2TB/s vs 1.8TB/s on 5090) and the ability to run FP16 without any quantization overhead.

---

## Implementation Order

1. **`gpu.py` fixes** — recommend_tier single-GPU FP16, test with CUDA_VISIBLE_DEVICES=0
2. **`flux_engine.py` verification** — confirm no-offload path on high-VRAM single GPU
3. **SeedVR2 subprocess verification** — single-GPU mode works
4. **Dockerfile** — build and test locally with `--gpus all`
5. **Weight download script** — verify all weights download correctly
6. **Cloud test** — rent A100 80GB, run end-to-end, collect benchmarks
7. **Documentation** — `CLOUD_DEPLOYMENT.md` with provider-specific instructions

---

## What NOT to Do

- No multi-GPU sharding (FSDP). Single-GPU A100/H100 has enough VRAM. FSDP adds complexity for no benefit in this scenario.
- No Kubernetes. Single-tenant, single-VM. K8s orchestration is for the future multi-tenant platform.
- No auto-scaling. One VM, one user, manual start/stop.
- No GPU sharing (MIG, time-slicing). Full GPU for the pipeline. Sharing introduces latency and VRAM fragmentation.
- No custom CUDA kernels or TensorRT optimization. PyTorch + native SDPA is the stack. Optimization is a future effort if benchmarks reveal specific bottlenecks.
