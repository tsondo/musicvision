> **Superseded by [`PLATFORM_SUPPORT_PLAN.md`](PLATFORM_SUPPORT_PLAN.md)** — see that document for
> the authoritative, file-organized view of all changes. This file is kept for history.

# Cloud Linux VM (A100 / H100 / H200) Support Plan

**Date:** 2026-02-26
**Scope:** Full pipeline on single- and multi-GPU cloud Linux VMs with A100, H100, or H200.
**Not in scope:** Multi-node inference, NVLink mesh beyond 2 GPUs, AMD/XPU.

---

## 1. Executive Summary

The codebase is CUDA-first, so A100/H100/H200 cloud VMs work largely out of the box.
Most required changes are in `gpu.py:recommend_tier()` (single high-VRAM GPU gap) and
`imaging/flux_engine.py` (unnecessary CPU offload on 80+ GB cards). Flash attention is
already optional (SDPA fallback exists). Total effort is **~2–3 hours**.

---

## 2. GPU Characteristics

| GPU | CC | VRAM | BF16 | FP8 native | FP8 tier? |
|---|---|---|---|---|---|
| A100 40 GB | 8.0 | 40 GB | ✓ | ✗ (CC < 8.9) | FP8 weights + BF16 compute |
| A100 80 GB | 8.0 | 80 GB | ✓ | ✗ | FP16 or FP8 storage |
| H100 80 GB | 9.0 | 80 GB | ✓ | ✓ | Native FP8 (e4m3fn, e5m2) |
| H200 141 GB | 9.0 | 141 GB | ✓ | ✓ | Native FP8 |

`_fp8_supported()` checks `>= (8, 9)` — correctly passes on H100/H200, correctly fails on A100.
FP8 tier loading on A100 stores weights in float8 but `FP8ScaledLinear.forward()` falls back
to BF16 compute — memory-efficient but not native FP8 speed. **This is correct behavior.**

---

## 3. Issues Found

### 3.1 `gpu.py:recommend_tier()` — high-VRAM single-GPU gap

**File:** `src/musicvision/utils/gpu.py`, lines 169–177

Current logic:
```python
if n_gpus >= 2 and primary_gb >= 24:
    return HumoTier.FP16        # ← only multi-GPU path to FP16
if primary_gb >= 20:
    return HumoTier.FP8_SCALED  # ← single A100 80GB / H100 80GB land here
```

A single A100 80 GB or H100 80 GB returns `FP8_SCALED` even though the full FP16 17B model
(~34 GB) plus encoders (~12 GB) fits comfortably in 80 GB. Same for H200 (141 GB).

**Fix:**
```python
if n_gpus >= 2 and primary_gb >= 24:
    return HumoTier.FP16
if n_gpus == 1 and primary_gb >= 48:
    return HumoTier.FP16   # single A100 80 GB / H100 80 GB / H200
if primary_gb >= 20:
    return HumoTier.FP8_SCALED
```

Threshold rationale: 34 GB (FP16 DiT) + 12 GB (T5 + VAE + Whisper) = 46 GB; 48 GB gives
adequate headroom for activation memory.

**Effort: 15 min.**

---

### 3.2 `gpu.py:recommend_tier()` — duplicate in `humo_engine.py` → consolidate

**File:** `src/musicvision/video/humo_engine.py`, lines 634–658

This function is a second, nearly identical copy of `recommend_tier()` defined at module
level in `humo_engine.py`. It has the same single-GPU FP16 gap and diverges silently
whenever `gpu.py:recommend_tier()` is updated.

**Fix (consolidation — do this once for both plans):**
1. Apply the `n_gpus == 1 and primary_gb >= 48 → FP16` fix to `gpu.py:recommend_tier()`.
2. Delete `humo_engine.py:recommend_tier()`.
3. Import from `gpu.py` at the top of `humo_engine.py`:
   ```python
   from musicvision.utils.gpu import recommend_tier  # single source of truth
   ```
4. `cli.py` and `api/app.py` already import from `gpu.py` — no change needed there.

The deduplication is a required part of this work, not optional cleanup. Having two copies
means tier logic will diverge again on the next change.

**Effort: 20 min (includes the fix in §3.1).**

---

### 3.3 `imaging/flux_engine.py` — FLUX high-VRAM strategy

**File:** `src/musicvision/imaging/flux_engine.py`

On a single A100 80 GB or H100 (≥28 GB free VRAM), `_select_strategy()` returns
`bf16_split`. But since `dit_device == encoder_device` on a single GPU, `load()` falls
through to `_load_bf16_offload()` which calls `enable_model_cpu_offload()` — this
unnecessarily moves T5-XXL to CPU between calls, adding latency on a GPU with 80 GB headroom.

**Fix:** Add a no-offload path when we have one large GPU:
```python
# In load():
if strategy == "bf16_split" and self.device_map.dit_device == self.device_map.encoder_device:
    self._pipe = self._load_bf16_no_offload(model_id, hf_token)
elif strategy in ("bf16_split", "bf16_offload"):
    self._pipe = self._load_bf16_offload(model_id, hf_token)
...

def _load_bf16_no_offload(self, model_id, token):
    """Single high-VRAM GPU: load everything on device, no offload needed."""
    import torch
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, token=token)
    pipe.to(self.device_map.primary)
    return pipe
```

**Effort: 45 min.**

---

### 3.4 `model_loader.py:FP16Loader` — misleading FSDP comment

**File:** `src/musicvision/video/model_loader.py`, lines 279–280

The docstring says *"On multi-GPU: uses FSDP to shard the DiT across both devices."*
The implementation places the entire DiT on `dit_device` (GPU0) with no sharding:
```python
model = model.to(dit_device)   # everything on GPU0
```

**Fix:** Update the docstring to accurately describe the behavior and state the limitation.
Do not attempt FSDP implementation in this plan (see §10).

**Effort: 5 min.**

---

### 3.5 Flash attention — already handled

`vendor/wan_dit_arch.py` imports flash_attn with `try/except`. When absent (common on
vanilla `torch` cloud images), the code automatically uses `F.scaled_dot_product_attention()`
— PyTorch's fused SDPA uses cuDNN internally on A100/H100/H200 (equivalent performance).
**No change needed.** ✓

If the cloud image has flash_attn (NVIDIA NGC, AWS DL AMI), it is used automatically.

---

### 3.6 `block_swap.py:teardown()` — unconditional CUDA call

**File:** `src/musicvision/video/block_swap.py`, line 136

```python
torch.cuda.empty_cache()   # unconditional
```

Works on cloud CUDA VMs but inconsistent with guarded style elsewhere. Fix shared with
MAC_SUPPORT_PLAN (applied once):
```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**Effort: 5 min.**

---

### 3.7 `wan_model.py:pre_blocks()` — hardcoded `'cuda'` autocast

**File:** `src/musicvision/video/wan_model.py`, line 129

```python
with torch.amp.autocast('cuda', dtype=torch.float32):
```

Works on A100/H100/H200 (CUDA devices). Fix is shared with MAC_SUPPORT_PLAN and should
be applied once (see PLATFORM_SUPPORT_PLAN.md §wan_model):
```python
import contextlib
_device_type = self.patch_embedding.weight.device.type
_ctx = (torch.amp.autocast(_device_type, dtype=torch.float32)
        if _device_type == "cuda" else contextlib.nullcontext())
with _ctx:
```

**Effort: 15 min (share with MPS plan).**

---

## 4. FP8 Behavior on A100 vs H100/H200

| GPU | `_fp8_supported()` | FP8ScaledLinear behavior |
|---|---|---|
| A100 (CC 8.0) | `False` | Weights stored as float8_e4m3fn; forward dequantizes to BF16 matmul. Memory: ~18 GB. Speed: slightly slower than native BF16. |
| H100/H200 (CC 9.0) | `True` | Native `torch._scaled_mm()` FP8 matmul. Memory: ~18 GB. Speed: fastest. |

For A100 40 GB, `recommend_tier()` → `FP8_SCALED` is correct: 18 GB DiT + 12 GB encoders
= 30 GB fits well. The BF16 compute fallback is a minor speed penalty, not a correctness issue.

For A100 80 GB and H100/H200, the new `n_gpus == 1 and primary_gb >= 48 → FP16` rule gives
better quality at full precision.

---

## 5. CUDA Version Requirements

| Component | Min CUDA | Notes |
|---|---|---|
| torch 2.5.1 | CUDA 11.8 or 12.1+ | All cloud A100/H100/H200 images have CUDA 12.x |
| `torch._scaled_mm` (FP8) | CUDA 12.0+ | Always present on H100/H200 cloud images |
| flash_attn (optional) | CUDA 11.6+ | Not required; SDPA fallback used when absent |
| BF16 tensor cores | CC 8.0+ | All three GPUs supported |

---

## 6. pyproject.toml — Installation Notes

No dependency version changes needed. Add a comment in the `[ml]` group:

```toml
# Cloud CUDA installation (A100/H100/H200):
#   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
#       --index-url https://download.pytorch.org/whl/cu124
#
# flash_attn is optional — install only if desired for peak attention throughput:
#   pip install flash-attn --no-build-isolation  (requires CUDA toolkit headers)
# Without it, PyTorch SDPA (cuDNN fused attention) is used automatically.
```

**Effort: 10 min.**

---

## 7. Summary — Changes Required

| File | Change | Effort | Priority |
|---|---|---|---|
| `gpu.py` | Single high-VRAM tier fix + consolidate duplicate `recommend_tier` | 20 min | **High** |
| `humo_engine.py` | Remove duplicate `recommend_tier`, import from `gpu.py` | 10 min | **High** |
| `imaging/flux_engine.py` | Add `_load_bf16_no_offload()` for single high-VRAM GPU | 45 min | Medium |
| `model_loader.py` | Fix FP16Loader docstring (remove false FSDP claim) | 5 min | Low |
| `block_swap.py` | Guard `torch.cuda.empty_cache()` | 5 min | Low |
| `wan_model.py` | Device-aware autocast | 15 min | Low |
| `pyproject.toml` | Add cloud install notes | 10 min | Low |
| **Total** | | **~1h 50 min** | |

---

## 8. Things That Already Work

| Concern | Status |
|---|---|
| `detect_devices()` — single GPU | ✓ `n_gpus == 1` path: all on `cuda:0`, offload to CPU |
| `detect_devices()` — 2-GPU cloud instance | ✓ Highest VRAM → DiT, second → encoders |
| Flash attention optional | ✓ `try/except` at import; SDPA fallback in `flash_attention()` |
| BF16 for T5 encoder on A100+ | ✓ A100 has native BF16 tensor cores |
| FP8 compute on H100/H200 | ✓ `_fp8_supported()` correctly gates on CC ≥ 8.9 |
| FP8 fallback on A100 (CC 8.0) | ✓ `FP8ScaledLinear` dequantizes to BF16 matmul |
| GGUF tiers on cloud | ✓ Float16 dequant via numpy; standard `F.linear` |
| SDPA attention on A100/H100 | ✓ PyTorch 2.x uses cuDNN fused attention via SDPA |
| VAE / Whisper on FP16 | ✓ FP16 supported on all three GPU types |
| Block swap on single GPU | ✓ `block_swap_count = 0` (default) keeps all blocks on GPU |
| `clear_vram()` | ✓ Uses try/except — safe on CUDA |
| `vram_info()` | ✓ Returns correct CUDA VRAM info |
| Assembly (ffmpeg) | ✓ CPU-based, no GPU dependency |

---

## 9. Recommended Cloud Configuration

| Cloud GPU | Tier | Block Swap | Notes |
|---|---|---|---|
| A100 40 GB (single) | `fp8_scaled` | 0 | 18 GB DiT + 12 GB encoders = 30 GB |
| A100 80 GB (single) | `fp16` | 0 | 34 GB DiT + 12 GB encoders = 46 GB |
| H100 80 GB (single) | `fp16` | 0 | Same as A100 80 GB |
| H200 141 GB (single) | `fp16` | 0 | Plenty of headroom |
| 2× A100 40 GB | `fp8_scaled` | 0 | DiT on GPU0, encoders on GPU1 |
| 2× A100 80 GB | `fp16` | 0 | DiT on GPU0 (34 GB), encoders on GPU1 |
| 2× H100 80 GB | `fp16` | 0 | Same as 2× A100 80 GB |

---

## 10. FSDP Multi-GPU: Deferred, Not Blocking

`FP16Loader` claims FSDP in its docstring but does not implement it. The full FP16 DiT
(~34 GB) is placed entirely on GPU0.

**Why this is not a blocker for cloud support:**
- Single A100/H100/H200 ≥ 80 GB → `fp16` tier, all fits on GPU0 without sharding. ✓
- 2× A100 40 GB → `fp8_scaled` tier (18 GB DiT on GPU0, fits within 40 GB). ✓
- 2× A100 80 GB → `fp16` tier, 34 GB DiT on GPU0 (80 GB), encoders on GPU1. ✓

FSDP sharding becomes relevant only when someone needs FP16 quality across a pair of
40 GB GPUs. That use case is addressed by the fp8_scaled fallback today. Revisit as
part of the `sp_size` multi-GPU speed work from the original HuMo FSDP framing —
that work should be scoped and planned separately when the need arises.

**Estimated FSDP effort: 8–12h** (out of scope for this plan).
