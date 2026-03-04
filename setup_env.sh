#!/usr/bin/env bash
# MusicVision — WSL2/Linux environment setup (uv)
#
# Prerequisites:
#   - WSL2 with Ubuntu 22.04+ (or native Linux)
#   - NVIDIA driver + CUDA toolkit 12.8+ installed
#   - uv installed: curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh

set -euo pipefail

PYTHON_VERSION="3.11"
CUDA_VERSION="cu128"
TORCH_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"
TORCHAUDIO_VERSION="2.10.0"

echo "╔══════════════════════════════════════╗"
echo "║   MusicVision Environment Setup      ║"
echo "╚══════════════════════════════════════╝"

# --- Check uv ---

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
echo "✓ uv: $(uv --version)"

# --- Check system dependencies ---

if ! command -v ffmpeg &>/dev/null; then
    echo "⚠ ffmpeg not found. Installing..."
    sudo apt-get update && sudo apt-get install -y ffmpeg
fi
echo "✓ ffmpeg: $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"

if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -2)
    echo "✓ GPUs:"
    echo "$GPU_INFO" | while read line; do echo "    $line"; done
else
    echo "⚠ nvidia-smi not available (OK for assembly-only dev)"
fi

if command -v nvcc &>/dev/null; then
    echo "✓ CUDA: $(nvcc --version 2>&1 | grep release | awk '{print $6}' | tr -d ',')"
else
    echo "⚠ nvcc not in PATH — CUDA toolkit may not be installed"
    echo "  For WSL: https://developer.nvidia.com/cuda-downloads (select WSL-Ubuntu)"
fi

# --- Create venv and install ---

echo ""
echo "Creating Python ${PYTHON_VERSION} environment..."

# uv will download Python 3.11 if not present
uv venv --python ${PYTHON_VERSION}

echo ""
echo "Installing PyTorch ${TORCH_VERSION} (CUDA ${CUDA_VERSION})..."
uv pip install \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

echo ""
echo "Installing MusicVision (editable + all extras)..."
uv pip install -e ".[ml,dev]"

echo ""
echo "Installing Real-ESRGAN (video upscaler)..."
uv pip install realesrgan

echo ""
echo "Installing audio separator..."
pip install "audio-separator[gpu]" 2>/dev/null || echo "⚠ audio-separator install failed (onnxruntime conflict — install separately)"

echo ""
echo "flash_attn is optional (SDPA fallback available). Install if nvcc is present:"
if command -v nvcc &>/dev/null; then
    echo "  Installing flash_attn..."
    uv pip install flash-attn --no-build-isolation || echo "⚠ flash_attn build failed — SDPA will be used as fallback"
else
    echo "  Skipping (no nvcc). Install manually: pip install flash-attn --no-build-isolation"
fi

# --- .env ---

echo ""
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env from .env.example — edit it to set your tokens and paths"
else
    echo "✓ .env already exists"
fi

# --- Verify ---

echo ""
echo "Verifying..."
uv run python -c "
import torch
print(f'  Python:     {__import__(\"sys\").version.split()[0]}')
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
print(f'  GPU count:  {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'    GPU {i}: {p.name} ({p.total_mem / 1024**3:.1f} GB)')
try:
    import flash_attn
    print(f'  flash_attn: {flash_attn.__version__}')
except ImportError:
    print('  flash_attn: not installed (using PyTorch SDPA fallback)')
try:
    import realesrgan
    print(f'  realesrgan: {realesrgan.__version__}')
except ImportError:
    print('  realesrgan: not installed')
"

echo ""
echo "Running tests..."
uv run pytest tests/ -v --tb=short

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Setup complete!                                               ║"
echo "║                                                                ║"
echo "║  Activate:  source .venv/bin/activate                          ║"
echo "║  Or prefix: uv run <command>                                   ║"
echo "║                                                                ║"
echo "║  CLI:       uv run musicvision --help                          ║"
echo "║  Server:    uv run musicvision serve DIR                       ║"
echo "║  API docs:  http://localhost:8000/docs                         ║"
echo "║                                                                ║"
echo "║  External engines (separate venvs — see README for setup):     ║"
echo "║    HunyuanVideo-Avatar: set HVA_REPO_DIR in .env              ║"
echo "║    SeedVR2 upscaler:    set SEEDVR2_REPO_DIR in .env          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
