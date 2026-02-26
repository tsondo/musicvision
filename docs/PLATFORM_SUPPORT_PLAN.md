# Platform Support Plan

**Date:** 2026-02-26
**Covers:** Apple Silicon MPS (M-series Mac) · Cloud CUDA (A100 / H100 / H200) · General platform-agnostic hardening
**Source plans:** [`MAC_SUPPORT_PLAN.md`](MAC_SUPPORT_PLAN.md) · [`CLOUD_SUPPORT_PLAN.md`](CLOUD_SUPPORT_PLAN.md)

---

## Quick Reference

| Platform | Status | Blocking issues |
|---|---|---|
| Cloud CUDA (A100/H100/H200) | ~works today; 2 gaps in tier logic | Single-GPU FP16 threshold; FLUX unnecessary offload |
| Apple Silicon MPS | Does not work today | RoPE float64/complex128 (hard); FP8 + quanto (medium) |

**Total estimated effort:** ~12–15 hours
- Platform-agnostic fixes: ~2h (do first; benefit all platforms)
- Cloud-specific: ~2h
- MPS-specific: ~9–11h (includes RoPE fix testing)

---

## Implementation Order

Work in this sequence. Items 1–5 are platform-agnostic and unblock everything else.

| Step | File | Work | Effort | Platforms |
|---|---|---|---|---|
| 1 | `pyproject.toml` | Add `psutil` dep; add install comments | 15 min | MPS, Cloud |
| 2 | `gpu.py` | MPS detection + memory + tier; deduplicate `recommend_tier`; high-VRAM single-GPU fix | 3h | All |
| 3 | `humo_engine.py` | Device-aware seed; remove duplicate `recommend_tier` | 30 min | All |
| 4 | `block_swap.py` | Device-agnostic `empty_cache` | 10 min | All |
| 5 | `wan_t5.py` + `wan_vae.py` | Device-agnostic `empty_cache` (×2) | 20 min | All |
| 6 | `wan_model.py` | Device-aware autocast | 20 min | All |
| 7 | `vendor/wan_dit_arch.py` | `_AmpCompat` fix; RoPE float64→float32 (Stage 1) | 1.5h | MPS |
| 8 | `model_loader.py` | T5 dtype on MPS; block FP8 on MPS; fix FP16Loader docstring | 1.5h | MPS, Cloud |
| 9 | `imaging/flux_engine.py` | MPS memory; no-offload high-VRAM; no-quantization on MPS | 3h | MPS, Cloud |
| 10 | **Smoke test (Cloud)** | `test_gpu_pipeline.py --phase 1` on A100 or H100 | — | Cloud |
| 11 | **Smoke test (MPS Phase 1)** | `test_gpu_pipeline.py --phase 1` on Apple Silicon | — | MPS |
| 12 | **Smoke test (MPS Phase 2)** | `test_gpu_pipeline.py --tier preview --phase 2 --steps 5` on Apple Silicon | — | MPS |
| 13 | If MPS Phase 2 fails at RoPE | Implement real-valued rotation fallback (§vendor/wan_dit_arch.py) | +2h | MPS |

---

## Changes by File

Each file section lists all platform changes together with verified source line references.

---

### `pyproject.toml`

**Platforms:** MPS + Cloud

**Changes:**
- Add `"psutil>=5.9"` to core `dependencies[]`. Required for `torch.mps` memory detection
  (`psutil.virtual_memory()`) since MPS has no `get_device_properties()` equivalent. Also
  useful for cloud diagnostics.
- Add installation comments in the `[ml]` optional group:

```toml
# --- Platform-specific installation ---
# Apple Silicon Mac (MPS):
#   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
#   (Do NOT use --index-url cu124 on Mac — that index has no MPS wheels)
#
# Cloud CUDA (A100/H100/H200):
#   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
#       --index-url https://download.pytorch.org/whl/cu124
#
# flash_attn is optional on all platforms. Install separately for peak attention speed:
#   pip install flash-attn --no-build-isolation  (CUDA toolkit headers required)
# Without it, PyTorch SDPA is used automatically (equivalent speed on cloud GPUs).
```

**Effort: 15 min**

---

### `src/musicvision/utils/gpu.py`

**Platforms:** MPS (major) + Cloud (minor)
**Verified source:** Lines 44–198 read in full.

#### `detect_devices()` — MPS branch (lines 67–121)

Add MPS detection **before** the CUDA checks. MPS is a single logical device — no
multi-GPU concept. All pipeline components (DiT, T5, VAE, Whisper) share the same MPS device.

```python
def detect_devices() -> DeviceMap:
    import torch

    # Apple Silicon — check MPS before CUDA
    if torch.backends.mps.is_available():
        mps = torch.device("mps")
        cpu = torch.device("cpu")
        log.info("Apple Silicon MPS detected — single-device mode (DiT + encoders on mps)")
        return DeviceMap(
            dit_device=mps,
            encoder_device=mps,
            vae_device=mps,
            offload_device=cpu,
        )

    # ... existing CUDA logic unchanged below ...
```

#### `clear_vram()` — device-agnostic cache clearing (lines 138–149)

Current code calls `torch.cuda.empty_cache()` unconditionally inside a `try/except`.
Replace the try block body:

```python
def clear_vram() -> None:
    import gc, torch
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    log.info("VRAM cleared")
```

#### `recommend_tier()` — deduplication + MPS branch + high-VRAM cloud fix (lines 152–177)

**Three changes in one function:**

1. **Deduplication:** `humo_engine.py` has an identical copy at lines 634–658. The canonical
   version lives here in `gpu.py`. The duplicate in `humo_engine.py` is deleted and replaced
   with `from musicvision.utils.gpu import recommend_tier`.

2. **Cloud gap:** Single GPU ≥ 48 GB (A100 80 GB, H100 80 GB, H200) never reached the FP16
   tier because the `n_gpus >= 2` guard was required. Fixed with a new single-GPU threshold.

3. **MPS branch:** System RAM–based tier selection, gated to preview-only for initial release.

```python
def recommend_tier(device_map: DeviceMap) -> "HumoTier":
    import torch
    from musicvision.models import HumoTier

    # Apple Silicon MPS — RAM-based, preview-only for initial release
    if device_map.dit_device.type == "mps":
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / 1024**3
        except ImportError:
            ram_gb = 16.0
        log.info("MPS device — recommending preview tier (initial release scope)")
        # Unlock GGUF tiers here in Phase 2 after smoke test validation:
        #   if ram_gb >= 32: return HumoTier.GGUF_Q4
        #   if ram_gb >= 48: return HumoTier.GGUF_Q6
        return HumoTier.PREVIEW

    # CUDA path
    try:
        primary_gb = torch.cuda.get_device_properties(device_map.dit_device).total_memory / 1024**3
        n_gpus = torch.cuda.device_count()
    except Exception:
        log.warning("CUDA not available — recommending preview tier")
        return HumoTier.PREVIEW

    if n_gpus >= 2 and primary_gb >= 24:
        return HumoTier.FP16
    if n_gpus == 1 and primary_gb >= 48:   # ← NEW: single A100 80GB / H100 / H200
        return HumoTier.FP16
    if primary_gb >= 20:
        return HumoTier.FP8_SCALED
    if primary_gb >= 16:
        return HumoTier.GGUF_Q6
    if primary_gb >= 12:
        return HumoTier.GGUF_Q4
    return HumoTier.PREVIEW
```

#### `vram_info()` — MPS branch (lines 180–198)

```python
def vram_info() -> list[dict]:
    import torch

    if torch.backends.mps.is_available():
        try:
            import psutil
            mem = psutil.virtual_memory()
            allocated = torch.mps.current_allocated_memory() / 1024**3
            return [{
                "index": 0,
                "name": "Apple Silicon (MPS)",
                "total_gb": round(mem.total / 1024**3, 1),
                "allocated_gb": round(allocated, 1),
                "free_gb": round(mem.available / 1024**3, 1),
                "compute_capability": "mps",
            }]
        except Exception:
            return [{"index": 0, "name": "Apple Silicon (MPS)", "compute_capability": "mps"}]

    # ... existing CUDA loop unchanged ...
```

#### `log_vram_usage()` — MPS branch

Add before the CUDA loop:
```python
if torch.backends.mps.is_available():
    allocated = torch.mps.current_allocated_memory() / 1024**3
    driver   = torch.mps.driver_allocated_memory() / 1024**3
    log.info("MPS (Apple Silicon): %.1f GB allocated, %.1f GB driver", allocated, driver)
    return
```

**Effort: 3h** (MPS detection is non-trivial; CUDA paths are untouched)

---

### `src/musicvision/video/humo_engine.py`

**Platforms:** All
**Verified source:** Lines 500–502 (`manual_seed`); lines 634–658 (duplicate `recommend_tier`).

#### Device-aware RNG seed (lines 500–502)

```python
# Replace:
if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)   # ← CUDA-only

# With:
if seed is not None:
    torch.manual_seed(seed)
    if dit_device.type == "cuda":
        torch.cuda.manual_seed(seed)
    elif dit_device.type == "mps":
        torch.mps.manual_seed(seed)
```

#### Remove duplicate `recommend_tier` (lines 634–658)

Delete the entire module-level `recommend_tier()` function. Add to imports at top of file:
```python
from musicvision.utils.gpu import recommend_tier  # single source of truth
```

**Effort: 30 min**

---

### `src/musicvision/video/block_swap.py`

**Platforms:** All
**Verified source:** Line 136 — `torch.cuda.empty_cache()` called unconditionally in `teardown()`.

```python
# Line 136 — replace:
torch.cuda.empty_cache()

# With:
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

**Effort: 10 min**

---

### `src/musicvision/video/wan_t5.py`

**Platforms:** MPS
**Verified source:** Lines 130–131 — guarded but MPS branch missing.

```python
# Lines 130–131 in unload() — replace:
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# With:
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

**Effort: 10 min**

---

### `src/musicvision/video/wan_vae.py`

**Platforms:** MPS
**Verified source:** Lines 145–146 — same pattern as wan_t5.py.

Same fix: add `elif torch.backends.mps.is_available(): torch.mps.empty_cache()`.

**Effort: 10 min**

---

### `src/musicvision/video/wan_model.py`

**Platforms:** All (correctness fix on Cloud; required fix on MPS)
**Verified source:** Line 129 — `torch.amp.autocast('cuda', dtype=torch.float32)`.

Also lines 113–114: `self.freqs = self.freqs.to(device)` — moves the complex128 RoPE
frequency tensor to the active device. On MPS this fails until the RoPE fix in
`wan_dit_arch.py` changes it to complex64.

```python
# Line 129 — replace:
with torch.amp.autocast('cuda', dtype=torch.float32):

# With:
import contextlib
_device_type = self.patch_embedding.weight.device.type
_ctx = (torch.amp.autocast(_device_type, dtype=torch.float32)
        if _device_type == "cuda" else contextlib.nullcontext())
with _ctx:
```

The float32 autocast prevents bfloat16 accumulation in sinusoidal timestep embeddings.
On MPS/CPU the computation stays in float32 naturally (no cast needed, no ctx needed).

**Effort: 20 min**

---

### `src/musicvision/video/vendor/wan_dit_arch.py`

**Platforms:** MPS (both changes); All (autocast change is a correctness fix for all)
**Verified source:** Lines 26–33 (`_AmpCompat`); lines 154–199 (`sinusoidal_embedding_1d`,
`rope_params`, `rope_apply`); lines 529, 537, 547, 576 (`amp.autocast` calls in blocks).

#### `_AmpCompat` class — device-aware autocast (lines 27–33)

`_AmpCompat.autocast()` is used as `@amp.autocast(enabled=False)` on `rope_params` and
`rope_apply` (lines 168, 179) and as `with amp.autocast(dtype=torch.float32):` inside
`WanAttentionBlock.forward` (lines 529, 537, 547) and `Head.forward` (line 576).

The `enabled=False` decorator use is safe on all devices (it's a no-op). The
`dtype=torch.float32` context manager use is what requires the CUDA guard.

```python
import contextlib as _contextlib

class _AmpCompat:
    @staticmethod
    def autocast(enabled=True, dtype=None):
        if not enabled:
            return _contextlib.nullcontext()
        # torch.amp.autocast with dtype override is reliable only on CUDA.
        # On MPS/CPU the computations are already float32 in those call sites.
        try:
            device_type = torch.get_default_device().type
        except AttributeError:
            device_type = "cpu"
        if device_type != "cuda":
            return _contextlib.nullcontext()
        if dtype is not None:
            return torch.amp.autocast(device_type, enabled=enabled, dtype=dtype)
        return torch.amp.autocast(device_type, enabled=enabled)
```

#### RoPE float64 / complex128 — hard MPS blocker (lines 154–199)

**Root cause confirmed by reading the source:**

- `sinusoidal_embedding_1d`, line 157: `position.type(torch.float64)` — MPS has no float64
- `rope_params`, lines 172–176: `torch.arange(...).to(torch.float64)` + `torch.polar()` →
  produces **complex128** (`torch.complex128`). MPS does not support complex128.
- `WanModel.__init__`, lines 696–700: `self.freqs = torch.cat([rope_params(...)])` stores
  complex128 as an instance attribute (not a registered buffer).
- `_WanModel.forward`, line 721: `self.freqs = self.freqs.to(device)` — moving complex128
  to MPS raises `RuntimeError` at the first denoising step.
- `WanModel.pre_blocks`, line 113: same `.to(device)` call — same failure.
- `rope_apply`, line 188: `.to(torch.float64)` on MPS tensor; `view_as_complex` on float64
  creates complex128.

**Stage 1 fix — float32 downcast** (apply first; test before considering Stage 2):

```python
# sinusoidal_embedding_1d, line 157 — replace:
position = position.type(torch.float64)
# With:
position = position.float()   # float32 precision is sufficient (output is cast to float32 anyway)

# rope_params, lines 172–174 — replace:
1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
# With:
1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(dim))
# torch.polar(ones_float32, freqs_float32) now produces complex64 (supported on MPS for storage)

# rope_apply, line 188 — replace:
x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
# With:
x_i = torch.view_as_complex(x[i, :seq_len].float().reshape(seq_len, n, -1, 2).contiguous())
# → view_as_complex on float32 produces complex64
```

After Stage 1: `self.freqs` is complex64. Moving complex64 to MPS is supported in
PyTorch 2.5+. Whether the complex64 multiply `x_i * freqs_i` in `rope_apply` works on
MPS is **unconfirmed** — it requires running the smoke test (implementation order step 12).

**Stage 2 fix — real-valued rotation** (only if Stage 1 fails with a complex-arithmetic error):

```python
# Replace all of rope_apply's inner loop with:
for i, (f, h, w) in enumerate(grid_sizes.tolist()):
    seq_len = f * h * w
    # freqs_i: [seq_len, 1, C/2] complex64 → split into cos/sin
    freqs_i = torch.cat([
        freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(seq_len, 1, -1)   # [seq, 1, C/2] complex64
    cos = freqs_i.real.float()           # [seq, 1, C/2]
    sin = freqs_i.imag.float()
    xi = x[i, :seq_len].float()          # [seq, N, C]
    x_even = xi[..., 0::2]              # [seq, N, C/2]
    x_odd  = xi[..., 1::2]
    x_rot = torch.stack([
        x_even * cos - x_odd  * sin,     # real part
        x_even * sin + x_odd  * cos,     # imag part
    ], dim=-1).flatten(-2)              # [seq, N, C]
    x_i = torch.cat([x_rot.to(x.dtype), x[i, seq_len:]])
    output.append(x_i)
```

This uses only real tensor ops (float32) — fully supported on MPS.

**Effort: 1.5h** (Stage 1 only) or **3.5h** (if Stage 2 is also needed)

---

### `src/musicvision/video/model_loader.py`

**Platforms:** MPS (T5 dtype; FP8 block) + Cloud (FP16 docstring)
**Verified source:** `_load_t5()` line 197 (bfloat16); `_load_fp8_dit()` general structure;
`FP16Loader` docstring lines 279–280; `get_loader()` lines 747–759.

#### T5 encoder dtype on MPS (`_load_t5()`, line 197)

BF16 is not hardware-accelerated on M1/M2 (no bfloat16 tensor cores before M3). Use FP16
on MPS for broad compatibility; keep BF16 on CUDA.

```python
# Replace:
encoder = WanT5Encoder(device=device, dtype=torch.bfloat16)

# With:
_t5_dtype = torch.float16 if device.type == "mps" else torch.bfloat16
encoder = WanT5Encoder(device=device, dtype=_t5_dtype)
```

#### Block FP8 tier on MPS (`get_loader()`, line 747)

`torch.float8_e4m3fn` tensors cannot be placed on MPS. The FP8 loader moves the entire
model including float8 buffers via `model.to(dit_device)` — this raises `RuntimeError`.
Block it at the loader factory with a clear message.

```python
def get_loader(tier: HumoTier, device: "torch.device | None" = None) -> BaseHumoLoader:
    """Return the correct loader for *tier*.

    Args:
        tier:   Requested precision tier.
        device: The dit_device for the model (used for platform compatibility checks).
                Pass device_map.dit_device when calling from the engine.
    """
    import torch
    if device is not None and device.type == "mps":
        if tier == HumoTier.FP8_SCALED:
            raise ValueError(
                f"FP8_SCALED tier is not supported on MPS: torch.float8_e4m3fn "
                f"tensors cannot be placed on MPS devices. "
                f"Use GGUF_Q4, GGUF_Q6, or PREVIEW instead."
            )
    match tier:
        ...  # existing cases unchanged
```

#### FP16Loader docstring — remove false FSDP claim (lines 279–280)

The docstring says *"On multi-GPU: uses FSDP to shard the DiT across both devices."*
The implementation loads the entire DiT onto `dit_device`. Update to say:

```
"Note: the full DiT is loaded onto dit_device (GPU0). No FSDP sharding is implemented.
On 2× 40 GB setups use fp8_scaled tier instead. See PLATFORM_SUPPORT_PLAN.md §10."
```

**Effort: 1.5h**

---

### `src/musicvision/imaging/flux_engine.py`

**Platforms:** MPS (memory detection; no quantization) + Cloud (no-offload high-VRAM path)
**Verified source:** `_free_vram_gb()` lines 292–301; `_select_strategy()` lines 304–327;
`_supports_fp8()` lines 330–343; `_pick_quant_type()` lines 346–363; `load()` lines 78–117;
`_load_split()` lines 202–216; `_load_bf16_offload()` lines 218–229.

#### `_free_vram_gb()` — MPS memory detection (lines 292–301)

Current code calls `torch.cuda.mem_get_info(device)` for any non-CPU device. On MPS this
raises, is caught by `except Exception`, and returns 0.0 — causing tier selection to always
pick `quantized_sequential`.

```python
def _free_vram_gb(device) -> float:
    """Return available memory in GB. Uses VRAM for CUDA, system RAM for MPS."""
    try:
        import torch
        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            return free_bytes / 1024**3
        if device.type == "mps":
            try:
                import psutil
                return psutil.virtual_memory().available / 1024**3
            except ImportError:
                return 16.0  # conservative fallback
    except Exception:
        pass
    return 0.0
```

#### `load()` — MPS no-offload + high-VRAM cloud no-offload (lines 78–117)

With `_free_vram_gb()` fixed, a 32+ GB MPS device returns `bf16_split` from
`_select_strategy()`. But `_load_split()` tries to place models on separate CUDA devices,
and `_load_bf16_offload()` calls `enable_model_cpu_offload()` which is unnecessary when
memory is plentiful.

The same issue affects single A100/H100 (80+ GB free → `bf16_split` strategy → falls
through to `_load_bf16_offload()` → unnecessary CPU offload).

**New helper:**
```python
def _load_bf16_no_offload(self, model_id: str, token):
    """Load FLUX entirely on primary device — no CPU offload.
    Used when: MPS (single device always), or CUDA with ≥28 GB free on single GPU."""
    import torch
    from diffusers import FluxPipeline
    _dtype = torch.float16 if self.device_map.primary.type == "mps" else torch.bfloat16
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=_dtype, token=token)
    pipe.to(self.device_map.primary)
    return pipe
```

**Updated dispatch in `load()`:**
```python
primary = self.device_map.primary
free_gb = _free_vram_gb(primary)
strategy = _select_strategy(free_gb, self.config)

if primary.type == "mps":
    # MPS: always single-device, no quantization
    if strategy.startswith("quantized"):
        strategy = "bf16_offload"  # fall back to float16 offload if RAM is tight
    if strategy in ("bf16_split", "bf16_offload") and free_gb >= _TIER_A_GB:
        self._pipe = self._load_bf16_no_offload(model_id, hf_token)
    else:
        self._pipe = self._load_bf16_offload(model_id, hf_token)
elif strategy == "bf16_split" and self.device_map.dit_device == self.device_map.encoder_device:
    # Single high-VRAM CUDA GPU — load everything on device, skip CPU offload
    self._pipe = self._load_bf16_no_offload(model_id, hf_token)
elif strategy in ("bf16_split", "bf16_offload"):
    if self.device_map.dit_device != self.device_map.encoder_device:
        self._pipe = self._load_split(model_id, hf_token)
    else:
        self._pipe = self._load_bf16_offload(model_id, hf_token)
else:
    quant_type = _pick_quant_type(primary, self.config.quant)
    self._pipe = self._load_quantized(model_id, hf_token, quant_type, strategy)
```

#### `_pick_quant_type()` — block quantization on MPS (lines 346–363)

```python
def _pick_quant_type(device, quant: FluxQuant):
    if device.type == "mps":
        # optimum-quanto quantization kernels are CUDA-only as of 0.2.x
        return None  # caller must not use this result to call _load_quantized()
    from optimum.quanto import qfloat8, qint8
    ...existing logic unchanged...
```

**Effort: 3h**

---

## Platform Matrix — What Works Where

| Component | Cloud A100/H100/H200 | Apple Silicon MPS | Notes |
|---|---|---|---|
| `detect_devices()` | ✓ (unchanged) | After fix | MPS branch added |
| `recommend_tier()` | After fix (single-GPU FP16 threshold) | After fix (preview-only initial) | Deduplicated from 2 copies to 1 |
| Flash attention | ✓ optional (SDPA fallback) | ✓ not installed, SDPA used | No change needed |
| BF16 for T5/DiT | ✓ native on A100+ | ✓ M3+ native; M1/M2 use FP16 | T5 uses FP16 on MPS |
| FP8 (HuMo DiT) | ✓ native H100/H200; BF16 fallback A100 | ✗ blocked | Float8 not supported on MPS |
| FP8 (FLUX) | ✓ `_supports_fp8()` gates correctly | ✗ quanto is CUDA-only | No quantization on MPS |
| GGUF tiers | ✓ (dequant→FP16→F.linear) | After RoPE fix (Phase 2) | GGUF linear itself works on MPS |
| Preview 1.7B | ✓ | After RoPE fix (Phase 1 target) | Same arch, same fix needed |
| FP16 17B | ✓ single GPU ≥ 48 GB after fix | ✗ impractical (MPS overhead) | Even 128 GB M4 Max not viable |
| RoPE (complex ops) | ✓ float64 works on CUDA | After float32 fix + testing | Stage 1: complex64; Stage 2: real-valued |
| `sinusoidal_embedding` | ✓ float64 works on CUDA | After float32 fix | Precision loss negligible |
| Block swap | ✓ (CUDA device) | ✓ after `empty_cache` fix | `.to(device)` works on MPS |
| VAE encode/decode | ✓ | ✓ (F.scaled_dot_product_attention) | No changes needed |
| Whisper audio encoder | ✓ | ✓ FP16 on MPS | No changes needed |
| FLUX no-offload | After `_load_bf16_no_offload` | After `_load_bf16_no_offload` | Saves latency on large-VRAM hardware |
| Assembly (ffmpeg) | ✓ | ✓ | CPU-based; no GPU dependency |

---

## Common Patterns Applied Consistently

Three idioms are used in multiple files. They are:

### Device-agnostic cache clearing
```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()
```
Applied to: `gpu.py:clear_vram()`, `block_swap.py:teardown()`, `wan_t5.py:unload()`,
`wan_vae.py` internal helper.

### Device-aware RNG seeding
```python
torch.manual_seed(seed)           # CPU RNG always
if device.type == "cuda":
    torch.cuda.manual_seed(seed)
elif device.type == "mps":
    torch.mps.manual_seed(seed)
```
Applied to: `humo_engine.py:_denoise()`.

### Device-aware autocast
```python
import contextlib
_device_type = <weight>.device.type
_ctx = torch.amp.autocast(_device_type, dtype=torch.float32) if _device_type == "cuda" \
       else contextlib.nullcontext()
with _ctx:
    ...
```
Applied to: `wan_model.py:pre_blocks()` and `vendor/wan_dit_arch.py:_AmpCompat`.

---

## Deferred Work

### FSDP multi-GPU sharding for FP16Loader

`FP16Loader` claims FSDP in its docstring but places the full 34 GB DiT on GPU0. This is
**not blocking** today because:
- Single GPU ≥ 48 GB (A100 80 GB, H100, H200): FP16 fits without sharding.
- 2× A100 40 GB: Use `fp8_scaled` tier (18 GB on GPU0).
- 2× A100 80 GB / 2× H100: 34 GB DiT on GPU0 (80 GB) — fits fine.

Revisit as part of the `sp_size` multi-GPU speed work from the original HuMo codebase.
**Estimated: 8–12 hours.** Track separately.

### GGUF / larger tiers on MPS (Phase 2)

After the preview tier smoke test passes (implementation order step 12), unlock GGUF tiers
by updating `recommend_tier()` in `gpu.py`:
```python
# Uncomment in recommend_tier() MPS branch:
if ram_gb >= 48: return HumoTier.GGUF_Q6
if ram_gb >= 32: return HumoTier.GGUF_Q4
```
Test on M3/M4 Mac with ≥ 32 GB RAM. The GGUF dequantization path (`GGUFLinear._dequantize`
→ float16 → `F.linear`) works on MPS without changes.

### MLX / CoreML

MLX offers ~1.5–2× better throughput than MPS+PyTorch on M3. Requires full inference stack
rewrite. **Estimated: 3–5 weeks.** Out of scope.

CoreML requires static model export incompatible with dynamic frame count. Not recommended.
