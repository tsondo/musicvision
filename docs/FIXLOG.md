# FIXLOG — Pre-GPU Validation Against Upstream References

Date: 2025-02-25

## Summary

Validated `wan_model.py` and `audio_encoder.py` against upstream implementations
before first GPU test. Found **major structural mismatches** in both components
that would have prevented checkpoint loading. Both files have been fixed.

---

## 1. wan_model.py — Vendored from Phantom-video/HuMo

**Upstream references:**
- `Phantom-video/HuMo/humo/models/wan_modules/model_humo.py` (DiT + blocks)
- `Phantom-video/HuMo/humo/models/wan_modules/attention.py` (flash attn / SDPA)
- `Phantom-video/HuMo/humo/models/audio/audio_proj.py` (audio projection)

### Mismatches found (all critical — would cause checkpoint load failure):

| Component | Our Code (WRONG) | Upstream (CORRECT) |
|---|---|---|
| Self-attn layer names | `q_proj, k_proj, v_proj, out_proj` | `q, k, v, o` + `norm_q, norm_k` (WanRMSNorm) |
| QK normalization | None | WanRMSNorm applied to q and k after projection |
| FFN architecture | 3-layer SwiGLU (`fc1/fc2/fc3`) | 2-layer GELU (`nn.Sequential(Linear, GELU, Linear)`) |
| Block modulation | Per-block `nn.Linear(dim, 6*dim)` (`adaLN_modulation`) | Additive `nn.Parameter([1, 6, dim])` (`modulation`) + global `time_projection` |
| Time projection | None (each block projects independently) | Global `nn.Sequential(SiLU, Linear(dim, 6*dim))` shared across all blocks |
| Head structure | Flat: `head_norm`, `head_proj`, `head_modulation` (Linear) | `Head` submodule: `head.norm`, `head.head`, `head.modulation` (Parameter) |
| AudioProjModel layers | `fc1, fc2, fc3, norm` | `audio_proj_glob_1/2/3.layer`, `audio_proj_glob_norm.layer` (DummyAdapterLayer wrapping) |
| AudioProjModel output_dim | 5120 (model dim) | 1536 (separate audio token dim) |
| Audio cross-attn kv_dim | Same as model dim (5120) | Separate kv_dim=1536 (via `WanSelfAttentionSepKVDim`) |
| Text cross-attn | Has tanh gate parameter | No gate — simple residual `x + cross_attn(...)` |
| RoPE implementation | Real sin/cos pairs | Complex number multiplication (`torch.view_as_complex/real`) |
| RoPE freq split (14B, hd=128) | (16, 24, 24) — temporal/H/W pairs | (22, 21, 21) — `[c-2*(c//3), c//3, c//3]` |
| Normalization | `nn.LayerNorm(elementwise_affine=False)` | `WanLayerNorm` (float32 cast) + `WanRMSNorm` |
| norm3 affine | `elementwise_affine=False` | `elementwise_affine=True` when `cross_attn_norm=True` |
| Module names | `text_embed, time_embed, patch_embed` | `text_embedding, time_embedding, patch_embedding` |
| GELU variant | Default (`approximate='none'`) | `approximate='tanh'` |
| Modulation init | Zeros | `randn / sqrt(dim)` |

### Fix applied:

Created `vendor/wan_dit_arch.py` — vendored from upstream with:
- Flash attention with SDPA fallback (no hard `flash_attn` dependency)
- No `einops` dependency (manual reshape/permute)
- No `diffusers`, `common.distributed`, or Ulysses SP dependencies
- All state_dict keys match upstream exactly

Rewrote `wan_model.py` as a wrapper that imports from vendor and adds:
- `CONFIG_14B` / `CONFIG_1_7B` presets (with `in_dim=36` for HuMo I2V)
- `from_config()` factory with meta-device support
- `pre_blocks()` / `post_blocks()` for BlockSwapManager integration
- Batch-tensor `forward()` (pre-concatenated inputs)

Updated `humo_engine.py` block-swap calling convention to use `**block_kwargs` dict.
Updated `model_loader.py` GGUF key remapping to match new upstream key names.

---

## 2. audio_encoder.py — Band Boundaries Fixed

**Upstream reference:**
- `kijai/ComfyUI-WanVideoWrapper/HuMo/nodes.py` (`HuMoEmbeds.process()`)

### Mismatches found:

| Component | Our Code (WRONG) | Upstream (CORRECT) |
|---|---|---|
| Band 0 | layers 0–6 (7 layers) | layers 0–7 (8 layers) |
| Band 1 | layers 7–13 (7 layers) | layers 8–15 (8 layers) |
| Band 2 | layers 14–19 (6 layers) | layers 16–23 (8 layers) |
| Band 3 | layers 20–26 (7 layers) | layers 24–31 (8 layers) |
| Band 4 | layers 27–32 (6 layers) | layer 32 only (1 layer) |

Our non-uniform 7/7/6/7/6 split was wrong. Upstream uses uniform 8/8/8/8 + final layer.

### Fix applied:

Changed `_BAND_BOUNDARIES` from `[(0,7),(7,14),(14,20),(20,27),(27,33)]`
to `[(0,8),(8,16),(16,24),(24,32),(32,33)]`.

### Remaining notes (not bugs, just differences in approach):

- **Windowed Whisper processing:** Upstream processes audio in 750×640-sample
  chunks through the feature extractor and 3000-frame chunks through the encoder.
  Our code processes the full audio at once. This is fine for HuMo's max ~3.88s
  clips but could OOM for very long audio. Not a checkpoint-compat issue.

- **Window extraction:** The upstream `get_audio_emb_window()` function is
  actually commented out in kijai's `HuMoEmbeds.process()`, meaning windowing
  is done elsewhere in the sampling pipeline. Our `audio_encoder.py` does
  windowing inline, which is functionally equivalent. The window offset may
  differ slightly from upstream but this only affects inference quality, not
  checkpoint compatibility.

---

## 3. Verification

- All 89 existing tests pass (`uv run pytest tests/ -v`)
- Model instantiation on meta device confirmed (1583 state_dict keys)
- All 16 critical checkpoint key patterns verified present
- All 11 critical parameter shapes verified (patch_embedding, modulation, ffn,
  audio cross-attn kv, head, time_projection, audio_proj DummyAdapterLayer)
- No old key patterns remain (q_proj, fc1, adaLN, head_norm, etc.)

---

## Files Changed

| File | Action |
|---|---|
| `src/musicvision/video/vendor/wan_dit_arch.py` | **NEW** — vendored DiT architecture |
| `src/musicvision/video/vendor/__init__.py` | Updated docstring |
| `src/musicvision/video/wan_model.py` | **REWRITTEN** — wrapper over vendor |
| `src/musicvision/video/audio_encoder.py` | Band boundaries fixed |
| `src/musicvision/video/humo_engine.py` | Block-swap calling convention updated |
| `src/musicvision/video/model_loader.py` | GGUF key mapping updated |
| `FIXLOG.md` | This file |
