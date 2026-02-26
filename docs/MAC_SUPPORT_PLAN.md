> **Superseded by [`PLATFORM_SUPPORT_PLAN.md`](PLATFORM_SUPPORT_PLAN.md)** — see that document for
> the authoritative, file-organized view of all changes. This file is kept for history.

# Apple Silicon (MPS) Support Plan

**Date:** 2026-02-26
**Scope:** Full pipeline support on M-series Macs using `torch.backends.mps` (PyTorch MPS backend).
**Out of scope:** Intel Macs; CoreML export; MLX (see §6).
**Initial release scope:** Preview tier (1.7B) only — see §4 and §5 for rationale.

---

## 1. Executive Summary

The codebase is currently CUDA-only. Every device-dependent file needs an MPS branch — mostly
mechanical changes. The largest work items are `gpu.py` (full rewrite of device detection +
memory queries) and `imaging/flux_engine.py` (VRAM-based tier selection + quantization path).

A newly discovered **hard blocker** (§2.14) affects all tiers: the vendored RoPE implementation
uses `torch.float64` and `torch.complex128`, neither of which is supported on MPS. This must
be resolved before any HuMo tier can run on MPS. Initial Mac support is therefore scoped to
the Preview (1.7B) tier, validated first before unlocking GGUF tiers.

**Estimated total effort: ~10–13 hours** (single engineer, existing familiarity with codebase).

No architectural changes to the inference loop are required once the device-detection and
RoPE layers are fixed.

---

## 2. Issues Found (by file)

### 2.1 `src/musicvision/utils/gpu.py` — **Critical, entire file**

| Location | Issue |
|---|---|
| `detect_devices()` | Calls `torch.cuda.device_count()` — no MPS branch |
| `_gpu_sort_key()` | `torch.cuda.get_device_properties()` — CUDA-only helper |
| `log_vram_usage()` | `torch.cuda.*` APIs — no MPS equivalent |
| `clear_vram()` | `torch.cuda.empty_cache()` + `torch.cuda.synchronize()` |
| `recommend_tier()` | `torch.cuda.get_device_properties()` + `device_count()` — no MPS RAM-based path |
| `vram_info()` | `torch.cuda.get_device_properties()` — no MPS equivalent |

MPS is a single logical device (no concept of a "secondary GPU"). Memory queries go through
`psutil.virtual_memory()` or `torch.mps.current_allocated_memory()` — there is no
`get_device_properties()` equivalent.

### 2.2 `src/musicvision/video/humo_engine.py` — **Minor, 1 line + duplicate function**

| Line | Issue |
|---|---|
| 502 | `torch.cuda.manual_seed(seed)` — CUDA RNG only; MPS has `torch.mps.manual_seed()` |
| 634–658 | `recommend_tier()` duplicated from `gpu.py` — consolidate to single source of truth |

### 2.3 `src/musicvision/video/wan_model.py` — **Minor, 1 line**

| Line | Issue |
|---|---|
| 129 | `torch.amp.autocast('cuda', dtype=torch.float32)` — hardcoded device type |

### 2.4 `src/musicvision/video/vendor/wan_dit_arch.py` — **Minor, `_AmpCompat` class**

| Lines | Issue |
|---|---|
| 27–33 | `_AmpCompat.autocast()` hardcodes `'cuda'` in `torch.amp.autocast()`. On MPS this silently becomes a no-op in some PyTorch versions or raises in others. |

This is our own vendored code so we own the fix.

### 2.5 `src/musicvision/video/model_loader.py` — **Medium**

| Location | Issue |
|---|---|
| `_load_t5()` | `dtype=torch.bfloat16` — bfloat16 is not hardware-accelerated on M1/M2; float16 is the correct choice for broad MPS compatibility |
| `FP8ScaledLoader._load_fp8_dit()` | Moves `torch.float8_e4m3fn` tensors to device via `model.to(dit_device)`. MPS **does not support float8 dtype** — raises `RuntimeError` at placement time. Must be blocked. |
| `_fp8_supported()` | Already returns `False` for non-CUDA (catches exceptions). Works, but should be explicit. |
| `GGUFLinear._dequantize()` | Dequantizes to float16 on CPU then `.to(self.quant_data.device)`. Works on MPS because float16 is supported and numpy→tensor→MPS is a valid path. **No change needed.** |

### 2.6 `src/musicvision/video/block_swap.py` — **Minor, 1 line**

| Line | Issue |
|---|---|
| 136 | `torch.cuda.empty_cache()` in `teardown()` — CUDA-specific, unconditional call |

### 2.7 `src/musicvision/video/wan_t5.py` — **Minor, 1 line**

| Lines | Issue |
|---|---|
| 130–131 | `if torch.cuda.is_available(): torch.cuda.empty_cache()` in `unload()` — no MPS branch |

### 2.8 `src/musicvision/video/wan_vae.py` — **Minor, 1 line**

| Lines | Issue |
|---|---|
| 145–146 | `if torch.cuda.is_available(): torch.cuda.empty_cache()` — no MPS branch |

### 2.9 `src/musicvision/imaging/flux_engine.py` — **Medium-High**

| Location | Issue |
|---|---|
| `_free_vram_gb()` | Returns `0.0` for `device.type != "cuda"` → tier selection always picks `quantized_sequential` on MPS |
| `_supports_fp8()` | Returns `False` for non-CUDA (correct); make explicit for MPS. |
| `_pick_quant_type()` | `optimum-quanto` `qfloat8`/`qint8` kernels are CUDA-only as of 0.2.x. On MPS, `quantize(pipe.transformer)` will fail. Must skip quantization on MPS. |
| `_load_split()` | Multi-GPU split (Tier A with 2 GPUs) is not applicable to MPS — single device only. |
| `_load_bf16_offload()` / `_load_split()` | `torch.bfloat16` is unreliable on M1/M2; should use `float16` for MPS. |
| `load()` | No MPS-specific loading path. With `_free_vram_gb()` fixed, strategy selection should work, but quantized paths must be blocked. |

### 2.10 `pyproject.toml` — **Minor**

- `psutil` is not listed as a dependency — needed for system RAM detection on MPS.
- `torch==2.5.1` in `[ml]` group. Mac users must install from the default PyPI index (no CUDA index URL).
- No Mac-specific install instructions.

### 2.11 `audio_encoder.py` — **No changes needed**

Standard Whisper + tensor arithmetic. No CUDA-specific calls. ✓

### 2.12 `scheduler.py` — **No changes needed**

Pure math on whatever device the latents are on. ✓

### 2.13 `vendor/wan_dit_arch.py` — Flash attention

Flash attention (`flash_attn` / `flash_attn_interface`) is not available on MPS. The
`flash_attention()` helper already has an SDPA fallback:

```python
# Line 77: only uses flash if FLASH_ATTN_2/3_AVAILABLE
if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
    ...  # flash path
# Line 135: SDPA fallback — works on MPS since PyTorch 2.0
out = Fn.scaled_dot_product_attention(...)
```

On MPS both `FLASH_ATTN_2_AVAILABLE` and `FLASH_ATTN_3_AVAILABLE` will be `False`. **No change needed.** ✓

---

### 2.14 `vendor/wan_dit_arch.py` — RoPE float64 / complex128: **hard MPS blocker**

Confirmed by reading the source. All failures occur at the first denoising step.

| Location | Line(s) | Issue |
|---|---|---|
| `sinusoidal_embedding_1d()` | 157 | `position.type(torch.float64)` — MPS has no float64 support |
| `rope_params()` | 171–176 | `torch.arange(...).to(torch.float64)` then `torch.polar()` → produces **complex128** (`torch.complex128`) |
| `WanModel.__init__` | 696–700 | `self.freqs = torch.cat([rope_params(...)])` stores a complex128 tensor as instance attribute |
| `_WanModel.forward` | 721–722 | `self.freqs = self.freqs.to(device)` — moving complex128 to MPS raises `RuntimeError` (MPS does not support complex128) |
| `WanModel.pre_blocks` | 113–114 | Same `self.freqs.to(device)` call — same failure |
| `rope_apply()` | 188 | `.to(torch.float64)` on an MPS tensor; `torch.view_as_complex()` on a float64 input creates complex128 — fails on MPS |

**Impact:** MPS inference is completely blocked. The error fires before the first DiT forward
pass, when `pre_blocks()` tries to move `self.freqs` to the MPS device.

**Resolution path — two stages (try Stage 1 first):**

**Stage 1 — float32 downcast** (likely sufficient):
Change all `torch.float64` usage in `rope_params`, `rope_apply`, and
`sinusoidal_embedding_1d` to `torch.float32`. This makes `self.freqs` complex64
(`torch.complex128` → `torch.complex64`). PyTorch 2.5+ supports complex64 tensor
placement on MPS. Whether complex64 arithmetic (the `x_i * freqs_i` multiply in
`rope_apply`) works on MPS is **unconfirmed — requires runtime testing**.

The precision trade-off is acceptable: the original code's float64 RoPE is already
truncated to float32 at the output of `sinusoidal_embedding_1d(...).float()`, so
float64 accumulation inside those functions is unnecessary.

**Stage 2 — real-valued rotation fallback** (if Stage 1 fails):
If MPS does not support complex64 arithmetic, replace `view_as_complex / multiply /
view_as_real` with the equivalent real-valued rotation matrix:
```python
# Pure float32 — fully MPS-compatible
cos_freqs = freqs_i.real   # [seq, 1, C/2]
sin_freqs = freqs_i.imag
x_even = x[i, :seq_len, :, 0::2]
x_odd  = x[i, :seq_len, :, 1::2]
x_rot_even = x_even * cos_freqs - x_odd  * sin_freqs
x_rot_odd  = x_even * sin_freqs + x_odd  * cos_freqs
# interleave back and flatten
```

**Testing gate:** Run the smoke test (see §7 step 11) after Stage 1. Only proceed to
Stage 2 if you see a complex-arithmetic error at that step.

---

## 3. Concrete Changes Per File

### 3.1 `src/musicvision/utils/gpu.py`
**Effort: ~2.5h**

```
detect_devices():
  - Add MPS check BEFORE the CUDA checks:
      if torch.backends.mps.is_available():
          mps = torch.device("mps")
          cpu = torch.device("cpu")
          log.info("Apple Silicon MPS detected — single-device mode")
          return DeviceMap(dit_device=mps, encoder_device=mps,
                           vae_device=mps, offload_device=cpu)
  - Keep all existing CUDA logic unchanged below.

clear_vram():
  - Replace the try block:
      if torch.backends.mps.is_available():
          torch.mps.empty_cache()
      elif torch.cuda.is_available():
          torch.cuda.empty_cache()
          torch.cuda.synchronize()

recommend_tier():
  - Add MPS branch before CUDA branch:
      if device_map.dit_device.type == "mps":
          try:
              import psutil
              ram_gb = psutil.virtual_memory().total / 1024**3
          except ImportError:
              ram_gb = 16.0  # conservative default
          # Initial release: preview tier only (see §4)
          # Unlock GGUF tiers after preview smoke test passes
          return HumoTier.PREVIEW

vram_info():
  - Add MPS branch:
      if torch.backends.mps.is_available():
          try:
              import psutil; mem = psutil.virtual_memory()
              allocated = torch.mps.current_allocated_memory() / 1024**3
              return [{"index": 0, "name": "Apple Silicon (MPS)",
                       "total_gb": round(mem.total/1024**3, 1),
                       "allocated_gb": round(allocated, 1),
                       "free_gb": round(mem.available/1024**3, 1),
                       "compute_capability": "mps"}]
          except Exception:
              return [{"index": 0, "name": "Apple Silicon (MPS)",
                       "compute_capability": "mps"}]

log_vram_usage():
  - Add MPS branch using torch.mps.current_allocated_memory() /
    torch.mps.driver_allocated_memory().
```

Note: `_gpu_sort_key()` is only called from the CUDA multi-GPU path — no change needed.

---

### 3.2 `src/musicvision/video/humo_engine.py`
**Effort: ~30 min**

```python
# Line 500-502 — replace:
if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# With:
if seed is not None:
    torch.manual_seed(seed)
    if dit_device.type == "cuda":
        torch.cuda.manual_seed(seed)
    elif dit_device.type == "mps":
        torch.mps.manual_seed(seed)
```

Also: remove the module-level `recommend_tier()` at lines 634–658. Import and re-export
the one from `gpu.py` instead (see §3.8 in PLATFORM_SUPPORT_PLAN.md for the consolidation).

---

### 3.3 `src/musicvision/video/wan_model.py`
**Effort: ~30 min**

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

The float32 autocast prevents bfloat16 accumulation in sinusoidal embeddings; on MPS/CPU
the natural float32 precision is already sufficient.

---

### 3.4 `src/musicvision/video/vendor/wan_dit_arch.py`
**Effort: ~1.5h (includes RoPE fix + _AmpCompat)**

**`_AmpCompat` fix** (lines 27–33):
```python
import contextlib as _contextlib

class _AmpCompat:
    @staticmethod
    def autocast(enabled=True, dtype=None):
        if not enabled:
            return _contextlib.nullcontext()
        # Only CUDA supports autocast with dtype overrides reliably
        device_type = torch.get_default_device().type if hasattr(torch, 'get_default_device') else 'cpu'
        if device_type != 'cuda':
            return _contextlib.nullcontext()
        if dtype is not None:
            return torch.amp.autocast(device_type, enabled=enabled, dtype=dtype)
        return torch.amp.autocast(device_type, enabled=enabled)
```

**RoPE float64 → float32 (Stage 1 fix):**
```python
# sinusoidal_embedding_1d, line 157 — replace:
position = position.type(torch.float64)
# With:
position = position.float()  # float32; precision is sufficient (output is cast to float32 anyway)

# rope_params, lines 172–174 — replace:
1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
# With:
1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(dim))
# → torch.polar(ones_float32, freqs_float32) now produces complex64 (MPS-compatible storage)

# rope_apply, line 188 — replace:
x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
# With:
x_i = torch.view_as_complex(x[i, :seq_len].float().reshape(seq_len, n, -1, 2).contiguous())
# → view_as_complex on float32 produces complex64
```

If Stage 1 is insufficient (MPS complex64 arithmetic fails), implement the real-valued
fallback in `rope_apply` as described in §2.14.

---

### 3.5 `src/musicvision/video/model_loader.py`
**Effort: ~1.5h**

**T5 dtype on MPS:**
```python
# In _load_t5():
_t5_dtype = torch.float16 if device.type == "mps" else torch.bfloat16
encoder = WanT5Encoder(device=device, dtype=_t5_dtype)
```

**Block FP8 tier on MPS (in `get_loader()`):**
```python
def get_loader(tier: HumoTier, device: "torch.device | None" = None) -> BaseHumoLoader:
    import torch
    if device is not None and device.type == "mps":
        if tier == HumoTier.FP8_SCALED:
            raise ValueError(
                "FP8_SCALED tier is not supported on MPS: torch.float8_e4m3fn "
                "cannot be placed on MPS devices. Use GGUF_Q4, GGUF_Q6, or PREVIEW."
            )
    ...
```

---

### 3.6 `src/musicvision/video/block_swap.py`
**Effort: ~15 min**

```python
# Line 136 in teardown() — replace:
torch.cuda.empty_cache()
# With:
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

---

### 3.7 `src/musicvision/video/wan_t5.py`
**Effort: ~15 min**

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

---

### 3.8 `src/musicvision/video/wan_vae.py`
**Effort: ~15 min**

Same pattern as wan_t5.py (lines 145–146).

---

### 3.9 `src/musicvision/imaging/flux_engine.py`
**Effort: ~2.5h**

**Memory detection (`_free_vram_gb`):**
```python
def _free_vram_gb(device) -> float:
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
                return 16.0  # conservative default
    except Exception:
        pass
    return 0.0
```

**Block quantization on MPS:**
```python
def _pick_quant_type(device, quant: FluxQuant):
    if device.type == "mps":
        return None   # optimum-quanto is CUDA-only; signal no-quantization
    from optimum.quanto import qfloat8, qint8
    ...existing logic...
```

**dtype for FLUX on MPS** (all `_load_*` helpers):
```python
_flux_dtype = torch.float16 if primary.type == "mps" else torch.bfloat16
```

**Strategy on MPS:** Skip quantized tiers; force `bf16_offload` when primary is MPS.
`enable_model_cpu_offload()` works on MPS via accelerate ≥ 0.30.

---

### 3.10 `pyproject.toml`
**Effort: ~30 min**

- Add `"psutil>=5.9"` to core `dependencies`.
- Add Mac-specific install comment in `[ml]` group.

---

## 4. Initial Release Scope: Preview Tier Only

**For the initial MPS release, only the Preview (1.7B) tier is supported.**

Reasons:
1. The RoPE fix (§2.14) must be validated by running a real forward pass through the DiT.
   The 1.7B model provides that validation without requiring large weight downloads.
2. MPS memory management in PyTorch has significant overhead for large models (hundreds of
   small allocations, metal command buffer fragmentation). Even on M4 Max 128 GB unified
   memory, running the 17B model (34 GB in FP16) via MPS may hit allocation limits or be
   unacceptably slow due to CPU↔GPU sync overhead.
3. GGUF tiers require the same RoPE fix plus additional validation (GGUF dequant path).

| System RAM | Initial release tier | Future (Phase 2) tier |
|---|---|---|
| < 16 GB | `preview` | `preview` |
| 16–31 GB | `preview` | `gguf_q4` (after Phase 2 validation) |
| 32–47 GB | `preview` | `gguf_q4` / `gguf_q6` |
| 48–63 GB | `preview` | `gguf_q6` |
| ≥ 64 GB | `preview` | `gguf_q6` (FP16 17B remains unsupported on MPS) |

**FP8_SCALED and FP16 (17B) are permanently blocked on MPS** (float8 not supported; FP16 17B
is impractical due to MPS memory overhead even on the largest M-series chips).

**FLUX on MPS:** Uses float16 (not bfloat16), no quantization, CPU offload. Not gated on
Preview-only scope — FLUX runs on MPS independently of HuMo tier.

---

## 5. Blockers (must resolve before MPS can work)

| # | Blocker | Severity | Resolution |
|---|---|---|---|
| 1 | `rope_params` / `rope_apply` / `sinusoidal_embedding_1d`: float64 + complex128 | **Hard blocker (all tiers)** | Stage 1: float32 downcast; Stage 2: real-valued rotation if needed |
| 2 | `torch.float8_e4m3fn` cannot be placed on MPS | **Hard blocker (FP8 tier)** | Block FP8_SCALED tier in `get_loader()` |
| 3 | `optimum-quanto` quantization is CUDA-only | **Hard blocker for FLUX Tier C/D** | Skip quantization on MPS; force `bf16_offload` |
| 4 | `_free_vram_gb()` returns 0.0 on MPS | **Hard blocker for FLUX** | Fix with `psutil` / `torch.mps` path |
| 5 | `psutil` not in dependencies | Minor | Add to `pyproject.toml` |
| 6 | bfloat16 unreliable on M1/M2 | Minor | Use float16 for T5 and FLUX when `device.type == "mps"` |
| 7 | `_AmpCompat.autocast('cuda')` in vendored arch | Minor | Falls back to no-op on MPS; fix for correctness |

---

## 6. MLX and CoreML — Out of Scope

### MLX
~1.5–2× faster than MPS+PyTorch for transformer attention on M3. Requires full rewrite of
inference stack. No existing HuMo/WanModel MLX port. **Estimated: 3–5 weeks.** Defer to Phase 3.

### CoreML
Incompatible with dynamic frame count. Not recommended.

---

## 7. Implementation Order

1. `pyproject.toml` — add `psutil` (5 min)
2. `gpu.py` — MPS detection, memory, recommend_tier, deduplicate (2.5h)
3. `block_swap.py` — `empty_cache` fix (15 min)
4. `wan_t5.py` + `wan_vae.py` — `empty_cache` fixes (30 min)
5. `humo_engine.py` — `manual_seed` fix, remove duplicate `recommend_tier` (30 min)
6. `wan_model.py` — autocast fix (30 min)
7. `vendor/wan_dit_arch.py` — `_AmpCompat` + RoPE Stage 1 float32 downcast (1.5h)
8. `model_loader.py` — T5 dtype, FP8 block (1.5h)
9. `imaging/flux_engine.py` — MPS memory + no-quantization path (2.5h)
10. **Smoke test — Phase 1:** `python scripts/test_gpu_pipeline.py --phase 1` on Apple Silicon
    Mac. Verifies: MPS device detection, `recommend_tier()` returns `preview`, weight status
    check. Expected: passes with no CUDA errors.
11. **Smoke test — Phase 2:** `python scripts/test_gpu_pipeline.py --tier preview --phase 2
    --steps 5` on Apple Silicon Mac. Verifies: T5 encode → VAE encode → RoPE (complex64 path)
    → DiT forward → VAE decode → MP4 save. If Phase 2 fails with a complex-arithmetic error
    at RoPE, implement the real-valued fallback (§2.14 Stage 2) and re-run.
12. **GGUF validation (Phase 2 Mac support):** After smoke test passes, update
    `recommend_tier()` to unlock `gguf_q4` / `gguf_q6` based on system RAM thresholds.
    Test with `--tier gguf_q4` on an M3 Mac with ≥ 32 GB RAM.

**Total estimated wall time: 10–13 hours.**

---

## 8. Files NOT Changed

| File | Reason |
|---|---|
| `video/scheduler.py` | Pure math; device-agnostic |
| `video/audio_encoder.py` | No CUDA calls found; Whisper + tensor ops |
| `vendor/wan_vae_arch.py` | Uses `F.scaled_dot_product_attention` — works on MPS ✓ |
| `vendor/wan_t5_arch.py` | Standard PyTorch ops |
| `vendor/wan_tokenizers.py` | CPU-only tokenization |
| `assembly/` | ffmpeg-based; no ML device dependency |
| `intake/` | LLM + CPU audio; no ML device dependency |
| `api/` | FastAPI; no device dependency |
| `cli.py` | Delegates to engines; picks up new `detect_devices()` automatically |
