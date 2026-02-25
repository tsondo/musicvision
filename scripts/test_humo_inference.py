#!/usr/bin/env python3
"""
CPU unit tests for HuMo TIA inference components.

Run with: python scripts/test_humo_inference.py
No GPU required — uses tiny configs that fit on CPU.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import traceback

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {name}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# FlowMatchScheduler tests
# ---------------------------------------------------------------------------

def test_scheduler_sigmas():
    from musicvision.video.scheduler import FlowMatchScheduler
    sched = FlowMatchScheduler(num_inference_steps=10, shift=5.0)
    assert sched.sigmas[0].item() == 1.0, f"sigma[0] should be 1.0, got {sched.sigmas[0]}"
    assert sched.sigmas[-1].item() < 0.1, f"sigma[-1] should be small, got {sched.sigmas[-1]}"
    assert len(sched.sigmas) == 11, f"expected 11 sigmas, got {len(sched.sigmas)}"

def test_scheduler_step():
    from musicvision.video.scheduler import FlowMatchScheduler
    sched = FlowMatchScheduler(num_inference_steps=10, shift=5.0)
    z = torch.zeros(1, 16, 4, 4, 4)
    v = torch.ones(1, 16, 4, 4, 4)
    z_next = sched.step(v, z, 0)
    # sigma_next - sigma_curr < 0 (denoising direction), so z_next < z when v=1
    assert z_next.shape == z.shape

def test_scheduler_add_noise():
    from musicvision.video.scheduler import FlowMatchScheduler
    sched = FlowMatchScheduler(num_inference_steps=10, shift=5.0)
    clean = torch.zeros(1, 16, 4, 4, 4)
    noise = torch.ones(1, 16, 4, 4, 4)
    noisy = sched.add_noise(clean, noise, 0)
    # At step 0, sigma=1.0, so result should be pure noise
    assert torch.allclose(noisy, noise, atol=1e-4), f"at sigma=1, noisy should equal noise"


# ---------------------------------------------------------------------------
# WanModel architecture tests
# ---------------------------------------------------------------------------

def _tiny_config():
    return dict(dim=64, num_heads=4, num_layers=2, text_dim=64, ffn_dim=128)

def test_wan_model_import():
    from musicvision.video.wan_model import WanModel, CONFIG_14B, CONFIG_1_7B
    assert CONFIG_14B['dim'] == 5120
    assert CONFIG_1_7B['dim'] == 1536

def test_sinusoidal_embedding():
    from musicvision.video.wan_model import sinusoidal_embedding
    t = torch.tensor([0.5, 0.8])
    emb = sinusoidal_embedding(t, dim=64)
    assert emb.shape == (2, 64), f"Expected [2, 64], got {emb.shape}"

def test_rope_freqs():
    from musicvision.video.wan_model import WanRoPE
    freqs = WanRoPE.get_3d_freqs(dim=64, F=2, H=4, W=4)
    expected_len = 2 * 4 * 4
    assert freqs.shape[0] == expected_len, f"Expected [{expected_len}, ...], got {freqs.shape}"

def test_audio_proj_model():
    from musicvision.video.wan_model import AudioProjModel
    proj = AudioProjModel(seq_len=8, bands=5, channels=1280, intermediate_dim=64, output_dim=64, context_tokens=4)
    # [B, F, 8, 5, 1280]
    audio = torch.randn(1, 3, 8, 5, 1280)
    out = proj(audio)
    assert out.shape == (1, 3, 4, 64), f"Expected [1, 3, 4, 64], got {out.shape}"

def test_wan_model_forward_no_audio():
    from musicvision.video.wan_model import WanModel
    cfg = _tiny_config()
    model = WanModel(**cfg, humo_audio=False)
    model.eval()
    B, F, H, W = 1, 2, 8, 8
    x = torch.randn(B, 36, F, H, W)
    t = torch.tensor([0.5])
    text = torch.randn(B, 4, cfg['text_dim'])
    with torch.no_grad():
        out = model(x, t, text, audio_features=None)
    assert out.shape == (B, 16, F, H, W), f"Expected [{B}, 16, {F}, {H}, {W}], got {out.shape}"

def test_wan_model_forward_with_audio():
    from musicvision.video.wan_model import WanModel
    cfg = _tiny_config()
    model = WanModel(**cfg, humo_audio=True)
    model.eval()
    B, F, H, W = 1, 2, 8, 8
    x = torch.randn(B, 36, F, H, W)
    t = torch.tensor([0.5])
    text = torch.randn(B, 4, cfg['text_dim'])
    audio = torch.randn(B, F, 8, 5, 1280)
    with torch.no_grad():
        out = model(x, t, text, audio_features=audio)
    assert out.shape == (B, 16, F, H, W), f"Expected [{B}, 16, {F}, {H}, {W}], got {out.shape}"

def test_wan_model_pre_post_blocks():
    """Test that pre_blocks + manual block loop + post_blocks == forward()."""
    from musicvision.video.wan_model import WanModel
    cfg = _tiny_config()
    model = WanModel(**cfg, humo_audio=False)
    model.eval()
    B, F, H, W = 1, 2, 8, 8
    x = torch.randn(B, 36, F, H, W)
    t = torch.tensor([0.5])
    text = torch.randn(B, 4, cfg['text_dim'])

    with torch.no_grad():
        # Standard forward
        out_fwd = model(x, t, text, audio_features=None)

        # Split forward
        x_seq, time_emb, text_ctx, audio_proj_out, freqs, F_frames, h, w = model.pre_blocks(x, t, text, None)
        for i, block in enumerate(model.blocks):
            x_seq = block(x_seq, time_emb, text_ctx, audio_proj_out, freqs, F_frames)
        out_split = model.post_blocks(x_seq, time_emb, F_frames, h, w)

    assert out_fwd.shape == out_split.shape, \
        f"Shape mismatch: forward={out_fwd.shape}, split={out_split.shape}"
    assert torch.allclose(out_fwd, out_split, atol=1e-4), \
        f"Output mismatch between forward() and pre/post_blocks()"

def test_wan_model_from_config():
    from musicvision.video.wan_model import WanModel
    m14 = WanModel.from_config("14B")
    assert m14.dim == 5120
    m17 = WanModel.from_config("1_7B")
    assert m17.dim == 1536


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_groups = [
        ("FlowMatchScheduler", [
            ("sigma schedule length and bounds", test_scheduler_sigmas),
            ("step() returns correct shape", test_scheduler_step),
            ("add_noise() at sigma=1 is pure noise", test_scheduler_add_noise),
        ]),
        ("WanModel", [
            ("import and config dicts", test_wan_model_import),
            ("sinusoidal_embedding shape", test_sinusoidal_embedding),
            ("WanRoPE.get_3d_freqs shape", test_rope_freqs),
            ("AudioProjModel forward shape", test_audio_proj_model),
            ("forward() no audio", test_wan_model_forward_no_audio),
            ("forward() with audio", test_wan_model_forward_with_audio),
            ("pre_blocks/post_blocks == forward()", test_wan_model_pre_post_blocks),
            ("from_config()", test_wan_model_from_config),
        ]),
    ]

    total, passed = 0, 0
    for group_name, tests in test_groups:
        print(f"\n{group_name}:")
        for name, fn in tests:
            total += 1
            if test(name, fn):
                passed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
