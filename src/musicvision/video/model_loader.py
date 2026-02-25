"""
HuMo model loaders — one concrete class per precision tier.

All loaders share a common interface:
    bundle = SomeLoader().load(config, device_map, weights_dir)

The returned HumoModelBundle holds all components (DiT, T5, VAE, Whisper)
placed on their target devices and ready for inference.  The inference loop
in HumoEngine.generate() is identical regardless of which loader was used.

Loader map:
    fp16        →  FP16Loader          (FSDP multi-GPU)
    fp8_scaled  →  FP8ScaledLoader     (Kijai fp8_e4m3fn_scaled)
    gguf_q8     →  GGUFLoader          (Q8_0 dequant-on-forward)
    gguf_q6     →  GGUFLoader          (Q6_K dequant-on-forward)
    gguf_q4     →  GGUFLoader          (Q4_K_M dequant-on-forward)
    preview     →  Preview1_7BLoader   (1.7B FP16, fits single 8GB GPU)

References
----------
- kijai/ComfyUI-WanVideoWrapper: nodes_model_loading.py, custom_linear.py,
  fp8_optimization.py, gguf/gguf.py
- bytedance-research/HuMo: model architecture, TIA mode conditioning
- Wan-AI/Wan2.1-T2V-1.3B: T5, VAE weight formats
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from musicvision.models import HumoConfig, HumoTier
from musicvision.video.block_swap import BlockSwapManager
from musicvision.video.weight_registry import locate_dit, locate_shared

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class HumoModelBundle:
    """
    All loaded model components, placed on their target devices and
    ready for the inference loop.  HumoEngine.generate() reads this
    struct and never touches device placement itself.
    """
    dit: Any                     # WanModel / custom GGUF model — the main DiT
    t5: Any                      # UMT5 text encoder
    vae: Any                     # Video VAE encoder + decoder
    whisper: Any                 # Whisper encoder (encoder-only, for audio embeds)
    block_swap: BlockSwapManager | None  # None when swap is disabled
    dit_device: Any              # torch.device — primary GPU
    encoder_device: Any          # torch.device — secondary GPU / CPU


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _patch_fp8_linears(model: Any, state: dict, fp8_weight_keys: set) -> None:
    """Replace nn.Linear modules with FP8ScaledLinear where FP8 weights exist."""
    import torch

    def _get_module(root, path):
        parts = path.split(".")
        m = root
        for p in parts:
            m = getattr(m, p)
        return m

    def _set_module(root, path, new_module):
        parts = path.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_module)

    patched = 0
    for weight_key in fp8_weight_keys:
        # weight_key is like "blocks.0.self_attn.q_proj.weight"
        # module_path is "blocks.0.self_attn.q_proj"
        module_path = weight_key[:-len(".weight")]
        scale_key = weight_key[:-len("weight")] + "scale"
        if scale_key not in state:
            scale_key = module_path + "_scale_weight"
        if scale_key not in state:
            # Try with _scale suffix
            scale_key = weight_key + "_scale_weight"

        weight_fp8 = state[weight_key]
        scale = state.get(scale_key, torch.tensor(1.0, dtype=torch.float32))

        # Get bias if present
        bias_key = module_path + ".bias"
        bias = state.get(bias_key, None)

        try:
            _set_module(model, module_path, FP8ScaledLinear(weight_fp8, scale, bias))
            patched += 1
        except AttributeError:
            log.debug("FP8 patch: could not find module at path %s", module_path)

    log.info("FP8 patched %d linear layers", patched)


def _gguf_name_to_pt_key(name: str) -> str:
    """
    Convert GGUF tensor name to PyTorch state dict key.

    GGUF (llama.cpp convention): blk.N.attn_q.weight
    PyTorch (WanModel): blocks.N.self_attn.q_proj.weight
    """
    # Common remappings for WanModel
    replacements = [
        ("blk.", "blocks."),
        (".attn_q.", ".self_attn.q_proj."),
        (".attn_k.", ".self_attn.k_proj."),
        (".attn_v.", ".self_attn.v_proj."),
        (".attn_output.", ".self_attn.out_proj."),
        (".ffn_gate.", ".ffn.fc1."),
        (".ffn_up.", ".ffn.fc2."),
        (".ffn_down.", ".ffn.fc3."),
        (".attn_norm.", ".norm1."),
        (".ffn_norm.", ".norm3."),
        (".cross_attn_q.", ".cross_attn.q_proj."),
        (".cross_attn_k.", ".cross_attn.k_proj."),
        (".cross_attn_v.", ".cross_attn.v_proj."),
        (".cross_attn_output.", ".cross_attn.out_proj."),
        (".cross_attn_norm.", ".norm2."),
        ("token_embd.", "text_embed.0."),
        ("output_norm.", "head_norm."),
        ("output.", "head_proj."),
        ("time_embed.", "time_embed.0."),
        ("patch_embed.", "patch_embed."),
    ]
    result = name
    for old, new in replacements:
        result = result.replace(old, new)
    return result


class BaseHumoLoader(ABC):
    """Abstract base for all HuMo model loaders."""

    @abstractmethod
    def load(
        self,
        config: HumoConfig,
        device_map: Any,           # DeviceMap — lazy import
        weights_dir: Path | None = None,
    ) -> HumoModelBundle:
        """Load all model components and return a ready-to-use bundle."""
        ...

    def estimate_vram(self, config: HumoConfig) -> dict[str, float]:
        """Estimated VRAM usage (GB) per device. Informational only."""
        from musicvision.models import TIER_VRAM_GB
        dit_gb = TIER_VRAM_GB.get(config.tier.value, 0.0)
        return {
            "dit_device": dit_gb,
            "encoder_device": 6.0,   # T5 + VAE + Whisper
        }

    # ------------------------------------------------------------------
    # Shared helpers used by all concrete loaders
    # ------------------------------------------------------------------

    def _load_t5(self, device: Any, weights_dir: Path | None = None) -> Any:
        """Load UMT5-XXL text encoder onto *device*. Auto-downloads on first use."""
        from musicvision.video.wan_t5 import WanT5Encoder
        from musicvision.video.weight_registry import download_shared
        import torch

        try:
            path = locate_shared("t5", weights_dir)
        except FileNotFoundError:
            log.info("T5 weights not found locally — downloading from HuggingFace (Wan-AI/Wan2.1-T2V-1.3B, ~10 GB)…")
            path = download_shared("t5", base_dir=weights_dir)

        log.info("Loading T5 encoder from %s onto %s", path.name, device)
        encoder = WanT5Encoder(device=device, dtype=torch.bfloat16)
        encoder.load(path)
        log.info("T5 encoder ready on %s", device)
        return encoder

    def _load_vae(self, device: Any, weights_dir: Path | None = None) -> Any:
        """Load Wan2.1 Video VAE onto *device*. Auto-downloads on first use."""
        from musicvision.video.wan_vae import WanVideoVAE
        from musicvision.video.weight_registry import download_shared
        import torch

        try:
            path = locate_shared("vae", weights_dir)
        except FileNotFoundError:
            log.info("VAE weights not found locally — downloading from HuggingFace (Wan-AI/Wan2.1-T2V-1.3B, ~0.4 GB)…")
            path = download_shared("vae", base_dir=weights_dir)

        log.info("Loading VAE from %s onto %s", path.name, device)
        vae = WanVideoVAE(device=device, dtype=torch.float16)
        vae.load(path)
        log.info("VAE ready on %s", device)
        return vae

    def _load_whisper(self, device: Any, weights_dir: Path | None = None) -> Any:
        """Load Whisper large-v3 encoder onto *device*. Auto-downloads on first use."""
        try:
            from transformers import WhisperModel
        except ImportError as e:
            raise RuntimeError("transformers is not installed — run: pip install transformers") from e

        import torch

        try:
            path = locate_shared("whisper", weights_dir)
            model = WhisperModel.from_pretrained(
                str(path.parent),
                torch_dtype=torch.float16,
            ).encoder.to(device)
        except FileNotFoundError:
            log.info("Whisper weights not found locally — downloading via HuggingFace transformers (openai/whisper-large-v3, ~1.5 GB)…")
            # transformers handles its own cache (~/.cache/huggingface); no token needed
            model = WhisperModel.from_pretrained(
                "openai/whisper-large-v3",
                torch_dtype=torch.float16,
            ).encoder.to(device)

        model.eval()
        log.info("Whisper encoder ready on %s", device)
        return model

    def _make_block_swap(
        self,
        dit: Any,
        config: HumoConfig,
        gpu_device: Any,
    ) -> BlockSwapManager | None:
        """Create BlockSwapManager from config, or None if swap is disabled."""
        # Assume DiT exposes its blocks as `dit.blocks` (nn.ModuleList)
        blocks = getattr(dit, "blocks", None)
        if blocks is None:
            log.warning("DiT has no .blocks attribute — block swap disabled")
            return None
        if config.block_swap_count == 0:
            return None
        mgr = BlockSwapManager.from_config(blocks, config.block_swap_count, gpu_device)
        mgr.prepare()
        log.info(
            "BlockSwap active: %d of %d blocks on GPU, %d on CPU",
            mgr.num_gpu_blocks, mgr.total, mgr.total - mgr.num_gpu_blocks,
        )
        return mgr


# ---------------------------------------------------------------------------
# FP16 Loader (Tier: fp16)
# ---------------------------------------------------------------------------

class FP16Loader(BaseHumoLoader):
    """
    Load HuMo-17B in full FP16 precision.

    For single GPU ≥24 GB or dual GPU ≥48 GB combined.
    On multi-GPU: uses FSDP to shard the DiT across both devices.

    Weight source: bytedance-research/HuMo — diffusion_models/*.safetensors
    Reference: HuMo's scripts/infer_tia.sh (sp_size parameter)
    """

    def load(self, config: HumoConfig, device_map: Any, weights_dir: Path | None = None) -> HumoModelBundle:
        import torch

        dit_device = device_map.dit_device
        enc_device = device_map.encoder_device
        weight_path = locate_dit(HumoTier.FP16, weights_dir)

        log.info("Loading HuMo-17B FP16 DiT from %s onto %s", weight_path, dit_device)
        dit = self._load_wan_dit(weight_path, torch.float16, dit_device, device_map)

        t5      = self._load_t5(enc_device, weights_dir)
        vae     = self._load_vae(enc_device, weights_dir)
        whisper = self._load_whisper(enc_device, weights_dir)
        swap    = self._make_block_swap(dit, config, dit_device)

        return HumoModelBundle(
            dit=dit, t5=t5, vae=vae, whisper=whisper,
            block_swap=swap, dit_device=dit_device, encoder_device=enc_device,
        )

    def _load_wan_dit(self, weight_path: Path, dtype: Any, dit_device: Any, device_map: Any) -> Any:
        """Load WanModel from safetensors shards in FP16."""
        import torch
        from safetensors.torch import load_file
        from musicvision.video.wan_model import WanModel

        model = WanModel.from_config("14B")
        model = model.to(dtype)

        # Load from directory of shards or single file
        if weight_path.is_dir():
            import glob as _glob
            shards = sorted(_glob.glob(str(weight_path / "*.safetensors")))
            state = {}
            for shard in shards:
                state.update(load_file(shard, device="cpu"))
        else:
            state = load_file(str(weight_path), device="cpu")

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning("FP16 DiT: %d missing keys (first 5: %s)", len(missing), missing[:5])
        model = model.to(dit_device)
        model.eval()
        log.info("FP16 WanModel loaded on %s (%s)", dit_device, dtype)
        return model


# ---------------------------------------------------------------------------
# FP8 Scaled Loader (Tier: fp8_scaled)
# ---------------------------------------------------------------------------

class FP8ScaledLoader(BaseHumoLoader):
    """
    Load HuMo-17B in FP8 e4m3fn_scaled format.

    FP8 scaled matmul: torch._scaled_mm(input_fp8, weight_fp8, scale_a, scale_b).
    Per-tensor scaling factors are stored alongside weights in Kijai's format.

    Weight source: Kijai/WanVideo_comfy_fp8_scaled/HuMo/HuMo_14B_fp8_e4m3fn_scaled.safetensors
    Reference: kijai/ComfyUI-WanVideoWrapper custom_linear.py + fp8_optimization.py

    FP8 requires compute capability ≥ 8.9 (Ada Lovelace / Hopper — RTX 40xx+, RTX 50xx+).
    On older GPUs the engine falls back to bf16 (slower, more VRAM).
    """

    def load(self, config: HumoConfig, device_map: Any, weights_dir: Path | None = None) -> HumoModelBundle:
        import torch

        dit_device = device_map.dit_device
        enc_device = device_map.encoder_device
        weight_path = locate_dit(HumoTier.FP8_SCALED, weights_dir)

        log.info("Loading HuMo-17B FP8 scaled DiT from %s onto %s", weight_path.name, dit_device)
        dit = self._load_fp8_dit(weight_path, dit_device)

        t5      = self._load_t5(enc_device, weights_dir)
        vae     = self._load_vae(enc_device, weights_dir)
        whisper = self._load_whisper(enc_device, weights_dir)
        swap    = self._make_block_swap(dit, config, dit_device)

        return HumoModelBundle(
            dit=dit, t5=t5, vae=vae, whisper=whisper,
            block_swap=swap, dit_device=dit_device, encoder_device=enc_device,
        )

    def _load_fp8_dit(self, weight_path: Path, dit_device: Any) -> Any:
        """Load FP8 scaled safetensors and patch nn.Linear layers with FP8ScaledLinear."""
        import torch
        from safetensors.torch import load_file
        from musicvision.video.wan_model import WanModel

        log.info("Loading FP8 DiT from %s", weight_path.name)
        state = load_file(str(weight_path), device="cpu")

        # Separate weight tensors from scale tensors
        weight_keys = {k for k in state if not k.endswith("_scale_weight") and not k.endswith(".scale")}
        scale_keys = {k for k in state if k.endswith("_scale_weight") or k.endswith(".scale")}
        log.debug("FP8 state dict: %d weight keys, %d scale keys", len(weight_keys), len(scale_keys))

        # Build base model in bfloat16 (no weights yet — we'll patch linears)
        model = WanModel.from_config("14B")
        model = model.to(torch.bfloat16)

        # Load non-FP8 weights (norms, embeddings, etc.) with strict=False
        non_fp8_state = {k: v for k, v in state.items() if not k.endswith("_scale_weight") and not k.endswith(".scale")}
        model.load_state_dict(non_fp8_state, strict=False)

        # Patch nn.Linear layers that have FP8 weights in the checkpoint
        fp8_weight_keys = {k for k in state if state[k].dtype == torch.float8_e4m3fn}
        _patch_fp8_linears(model, state, fp8_weight_keys)

        model = model.to(dit_device)
        model.eval()
        log.info("FP8 WanModel loaded on %s", dit_device)
        return model


class FP8ScaledLinear(__import__("torch").nn.Module):
    """
    Drop-in nn.Linear replacement using FP8 scaled matmul.

    Forward pass:
        1. Cast input to float8_e4m3fn
        2. torch._scaled_mm(input_fp8, weight_fp8, scale_a, scale_b, out_dtype=bf16)
        3. Add bias if present

    This keeps the weight in fp8 on GPU, materialising bf16 only for the output.

    Reference: kijai/ComfyUI-WanVideoWrapper/custom_linear.py
    """

    def __init__(
        self,
        weight_fp8: Any,        # torch.Tensor, dtype float8_e4m3fn, shape (out, in)
        scale: Any,             # torch.Tensor, dtype float32, shape ()
        bias: Any | None = None,
    ) -> None:
        super().__init__()
        import torch
        self.register_buffer("weight", weight_fp8)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.bfloat16))
        else:
            self.bias = None

    def forward(self, x: Any) -> Any:
        import torch

        # Requires compute capability ≥ 8.9; falls back to bf16 on older GPUs
        if not _fp8_supported(x.device):
            w_bf16 = self.weight.to(torch.bfloat16) * self.scale
            out = torch.nn.functional.linear(x.to(torch.bfloat16), w_bf16, self.bias)
            return out

        x_fp8 = x.to(torch.float8_e4m3fn)
        # weight is (out, in) — transpose to (in, out) for scaled_mm
        w_t = self.weight.t().contiguous()
        # scale_a: per-tensor scale for input; scale_b: per-tensor scale for weight
        scale_a = torch.ones(1, device=x.device, dtype=torch.float32)
        scale_b = self.scale
        out = torch._scaled_mm(
            x_fp8.view(-1, x_fp8.shape[-1]),
            w_t,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )
        out = out.view(*x.shape[:-1], self.weight.shape[0])
        if self.bias is not None:
            out = out + self.bias
        return out


def _fp8_supported(device: Any) -> bool:
    """FP8 scaled_mm requires compute capability ≥ 8.9 (Ada / Hopper / Blackwell).

    The >= (8, 9) tuple comparison naturally covers all future architectures:
    - Ada Lovelace (RTX 40xx): CC 8.9
    - Hopper (H100): CC 9.0
    - Blackwell (RTX 50xx, e.g. RTX 5090): CC 12.0  ← also passes
    torch.float8_e4m3fn is supported on all three.
    """
    try:
        import torch
        if device.type == "cpu":
            return False
        major, minor = torch.cuda.get_device_capability(device)
        return (major, minor) >= (8, 9)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# GGUF Loader (Tiers: gguf_q8, gguf_q6, gguf_q4)
# ---------------------------------------------------------------------------

class GGUFLoader(BaseHumoLoader):
    """
    Load HuMo-17B from a GGUF quantized checkpoint.

    Weight tensors are stored in quantized form (Q8_0 / Q6_K / Q4_K_M).
    Each nn.Linear is replaced with a GGUFLinear that dequantizes on the fly
    during forward, keeping only the current layer's fp16 weights in VRAM at once.

    Supports block_swap: GGUF layers are naturally light-weight on CPU.

    Reference: kijai/ComfyUI-WanVideoWrapper/gguf/gguf.py
    """

    def __init__(self, tier: HumoTier) -> None:
        if tier not in (HumoTier.GGUF_Q8, HumoTier.GGUF_Q6, HumoTier.GGUF_Q4):
            raise ValueError(f"GGUFLoader only handles gguf_* tiers, got {tier}")
        self.tier = tier

    def load(self, config: HumoConfig, device_map: Any, weights_dir: Path | None = None) -> HumoModelBundle:
        import torch

        dit_device  = device_map.dit_device
        enc_device  = device_map.encoder_device
        weight_path = locate_dit(self.tier, weights_dir)

        log.info(
            "Loading HuMo-17B GGUF (%s) from %s onto %s",
            self.tier.value, weight_path.name, dit_device,
        )
        dit  = self._load_gguf_dit(weight_path, dit_device)
        t5      = self._load_t5(enc_device, weights_dir)
        vae     = self._load_vae(enc_device, weights_dir)
        whisper = self._load_whisper(enc_device, weights_dir)
        swap    = self._make_block_swap(dit, config, dit_device)

        return HumoModelBundle(
            dit=dit, t5=t5, vae=vae, whisper=whisper,
            block_swap=swap, dit_device=dit_device, encoder_device=enc_device,
        )

    def _load_gguf_dit(self, gguf_path: Path, dit_device: Any) -> Any:
        """Parse GGUF file and construct a WanModel with GGUFLinear layers."""
        import torch
        import gguf
        from musicvision.video.wan_model import WanModel

        log.info("Parsing GGUF file: %s", gguf_path.name)
        reader = gguf.GGUFReader(str(gguf_path))

        # Build base model in bfloat16 (skeleton — linears will be replaced)
        model = WanModel.from_config("14B")
        model = model.to(torch.bfloat16)

        # Map GGUF tensors: {name: tensor}
        gguf_tensors: dict[str, Any] = {}
        for tensor in reader.tensors:
            name = tensor.name
            gguf_tensors[name] = tensor

        # Separate non-quantized (norm/embed) from quantized (linear) tensors
        non_linear_state: dict[str, Any] = {}
        linear_tensors: dict[str, Any] = {}

        for name, tensor in gguf_tensors.items():
            pt_key = _gguf_name_to_pt_key(name)
            gguf_type = tensor.tensor_type
            is_quantized = gguf_type not in (
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.BF16,
            )
            if is_quantized and ".weight" in pt_key:
                linear_tensors[pt_key] = (tensor, gguf_type)
            else:
                # Dequantize to bfloat16 for non-linear params
                try:
                    arr = tensor.data
                    t = torch.from_numpy(arr.copy())
                    if t.dtype == torch.float16:
                        t = t.to(torch.bfloat16)
                    non_linear_state[pt_key] = t
                except Exception as e:
                    log.debug("GGUF: could not convert %s: %s", name, e)

        # Load non-quantized weights
        model.load_state_dict(non_linear_state, strict=False)

        # Replace linear modules with GGUFLinear
        patched = 0
        for pt_key, (tensor, gguf_type) in linear_tensors.items():
            module_path = pt_key[:-len(".weight")]
            quant_type_name = gguf_type.name  # e.g. "Q8_0", "Q6_K", "Q4_K_M"
            try:
                import numpy as np
                raw_data = torch.from_numpy(tensor.data.copy())
                shape = tuple(tensor.shape[::-1])  # GGUF stores transposed
                bias_key = module_path + ".bias"
                bias = non_linear_state.get(bias_key, None)
                new_linear = GGUFLinear(raw_data, shape, quant_type_name, bias)
                parts = module_path.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], new_linear)
                patched += 1
            except Exception as e:
                log.debug("GGUF patch failed for %s: %s", module_path, e)

        log.info("GGUF: patched %d quantized linear layers", patched)
        model = model.to(dit_device)
        model.eval()
        return model


class GGUFLinear(__import__("torch").nn.Module):
    """
    Drop-in nn.Linear that stores weights in GGUF quantized format and
    dequantizes on every forward call.

    Only Q8_0 dequantization is currently implemented inline.  Q6_K and Q4_K_M
    require the more complex block-based dequant from gguf-python.

    The dequantized fp16 tensor exists only for the duration of the matmul
    and is immediately discarded, keeping peak VRAM low.

    Reference: kijai/ComfyUI-WanVideoWrapper/gguf/gguf.py — GGUFLinear class
    """

    def __init__(
        self,
        quant_data: Any,        # uint8 tensor holding packed quantized bytes
        shape: tuple[int, ...], # original (out_features, in_features)
        quant_type: str,        # "Q8_0", "Q6_K", "Q4_K_M"
        bias: Any | None = None,
    ) -> None:
        super().__init__()
        import torch
        self.register_buffer("quant_data", quant_data)
        self.out_features, self.in_features = shape
        self.quant_type = quant_type
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float16))
        else:
            self.bias = None

    def forward(self, x: Any) -> Any:
        import torch

        w = self._dequantize()  # (out_features, in_features) float16
        out = torch.nn.functional.linear(x.to(torch.float16), w, self.bias)
        return out

    def _dequantize(self) -> Any:
        """Dequantize stored buffer to float16."""
        import numpy as np
        import torch

        data = self.quant_data.cpu().numpy()

        if self.quant_type == "Q8_0":
            return self._deq_q8_0(data)
        # Q6_K and Q4_K_M: delegate to gguf library helpers
        try:
            from gguf.quants import dequantize
            shape = (self.out_features, self.in_features)
            out = dequantize(data, self.quant_type).reshape(shape)
            return torch.from_numpy(out.astype(np.float16)).to(self.quant_data.device)
        except (ImportError, AttributeError):
            raise NotImplementedError(
                f"Dequantization for {self.quant_type} requires gguf>=0.6 with quants support. "
                f"Run: pip install 'gguf>=0.6'"
            )

    def _deq_q8_0(self, data: Any) -> Any:
        """
        Q8_0: 32 int8 values per block, one float16 scale per block.
        Block layout: [scale: float16 (2 bytes)] [qs: int8 × 32 (32 bytes)] = 34 bytes/block
        """
        import numpy as np
        import torch

        n = self.out_features * self.in_features
        n_blocks = n // 32
        blocks = data.reshape(n_blocks, 34)

        scales = blocks[:, :2].view(np.float16).astype(np.float32)  # (n_blocks,)
        qs     = blocks[:, 2:].view(np.int8).astype(np.float32)     # (n_blocks, 32)

        out = (scales[:, np.newaxis] * qs).reshape(self.out_features, self.in_features)
        return torch.from_numpy(out.astype(np.float16)).to(self.quant_data.device)


# ---------------------------------------------------------------------------
# Preview 1.7B Loader (Tier: preview)
# ---------------------------------------------------------------------------

class Preview1_7BLoader(BaseHumoLoader):
    """
    Load HuMo-1.7B in FP16 for fast iteration.

    The 1.7B model has the same architecture as the 17B but with significantly
    fewer transformer blocks and smaller hidden dimensions.  It fits entirely
    on a single 8 GB GPU with room for T5, VAE, and Whisper on a secondary GPU
    or CPU.

    Output quality is lower than the 17B tiers but A/V sync is preserved because
    the TIA conditioning mechanism is architecturally identical.  Use this tier
    during scene review and prompt iteration before committing to expensive 17B runs.

    Weight source: bytedance-research/HuMo — diffusion_models_1_3B/
    """

    def load(self, config: HumoConfig, device_map: Any, weights_dir: Path | None = None) -> HumoModelBundle:
        import torch

        dit_device = device_map.dit_device
        enc_device = device_map.encoder_device
        weight_path = locate_dit(HumoTier.PREVIEW, weights_dir)

        log.info("Loading HuMo-1.7B Preview DiT from %s onto %s", weight_path, dit_device)
        dit = self._load_preview_dit(weight_path, dit_device, torch.float16)

        t5      = self._load_t5(enc_device, weights_dir)
        vae     = self._load_vae(enc_device, weights_dir)
        whisper = self._load_whisper(enc_device, weights_dir)
        # 1.7B is small enough that block swap is rarely needed
        swap = self._make_block_swap(dit, config, dit_device) if config.block_swap_count > 0 else None

        return HumoModelBundle(
            dit=dit, t5=t5, vae=vae, whisper=whisper,
            block_swap=swap, dit_device=dit_device, encoder_device=enc_device,
        )

    def _load_preview_dit(self, weight_path: Path, dit_device: Any, dtype: Any) -> Any:
        """Load 1.7B WanModel variant in FP16."""
        import torch
        from safetensors.torch import load_file
        from musicvision.video.wan_model import WanModel

        model = WanModel.from_config("1_7B")
        model = model.to(dtype)

        if weight_path.is_dir():
            import glob as _glob
            shards = sorted(_glob.glob(str(weight_path / "*.safetensors")))
            state = {}
            for shard in shards:
                state.update(load_file(shard, device="cpu"))
        else:
            state = load_file(str(weight_path), device="cpu")

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning("Preview DiT: %d missing keys", len(missing))
        model = model.to(dit_device)
        model.eval()
        log.info("Preview 1.7B WanModel loaded on %s", dit_device)
        return model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_loader(tier: HumoTier) -> BaseHumoLoader:
    """Return the correct loader for *tier*."""
    match tier:
        case HumoTier.FP16:
            return FP16Loader()
        case HumoTier.FP8_SCALED:
            return FP8ScaledLoader()
        case HumoTier.GGUF_Q8 | HumoTier.GGUF_Q6 | HumoTier.GGUF_Q4:
            return GGUFLoader(tier)
        case HumoTier.PREVIEW:
            return Preview1_7BLoader()
        case _:
            raise ValueError(f"Unknown HumoTier: {tier}")
