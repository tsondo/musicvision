"""Compare our computed zero_vae against the official HuMo pre-computed one.

Key question: the upstream WanVAE.encode() takes input in [-1,1] range directly.
Our WanVideoVAE.encode() normalizes [0,1] -> [-1,1] (i.e., 0.0 becomes -1.0).
The original zero_vae was computed from torch.zeros() passed directly to WanVAE.encode(),
meaning input was 0.0 in [-1,1] range (mid-gray), NOT -1.0 (true black).
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

OFFICIAL_PATH = Path.home() / ".cache/musicvision/weights/humo/reference/zero_vae_129frame.pt"

# Load official (480p: 60x104 latent)
official = torch.load(OFFICIAL_PATH, map_location="cpu", weights_only=True)
print(f"Official: shape={official.shape}, dtype={official.dtype}")
print(f"  mean={official.float().mean():.6f}, std={official.float().std():.6f}")
print(f"  min={official.float().min():.6f}, max={official.float().max():.6f}")

# Load VAE
from musicvision.video.weight_registry import locate_shared
weights_dir = Path.home() / ".cache/musicvision/weights"
vae_path = locate_shared("vae", base_dir=weights_dir)
print(f"\nVAE path: {vae_path}")

from musicvision.video.vendor.wan_vae_arch import WanVAE
vae = WanVAE(vae_pth=str(vae_path), device="cpu")

H, W = 480, 832
n_frames = 129

# Test 1: Input = 0.0 (mid-gray in [-1,1] range) — what the original likely used
print(f"\n--- Test 1: Input = 0.0 (mid-gray, [-1,1] range) ---")
midgray = [torch.zeros(3, n_frames, H, W)]  # direct WanVAE format: list of [C,T,H,W]
with torch.no_grad():
    result_midgray = vae.encode(midgray)
lat_midgray = result_midgray[0]  # [16, 33, 60, 104]
print(f"  shape={lat_midgray.shape}")
print(f"  mean={lat_midgray.float().mean():.6f}, std={lat_midgray.float().std():.6f}")
diff_midgray = (lat_midgray.float() - official.float())
print(f"  Diff from official: abs_mean={diff_midgray.abs().mean():.6f}, abs_max={diff_midgray.abs().max():.6f}")

# Test 2: Input = -1.0 (true black in [-1,1] range) — what our engine computes
# (because we pass zeros in [0,1] which becomes -1 after normalization)
print(f"\n--- Test 2: Input = -1.0 (true black, [-1,1] range) ---")
black = [torch.full((3, n_frames, H, W), -1.0)]
with torch.no_grad():
    result_black = vae.encode(black)
lat_black = result_black[0]
print(f"  shape={lat_black.shape}")
print(f"  mean={lat_black.float().mean():.6f}, std={lat_black.float().std():.6f}")
diff_black = (lat_black.float() - official.float())
print(f"  Diff from official: abs_mean={diff_black.abs().mean():.6f}, abs_max={diff_black.abs().max():.6f}")

# Which is closer?
print(f"\n--- Summary ---")
print(f"Mid-gray (0.0) diff: mean={diff_midgray.abs().mean():.6f}, max={diff_midgray.abs().max():.6f}")
print(f"Black (-1.0)   diff: mean={diff_black.abs().mean():.6f}, max={diff_black.abs().max():.6f}")

if diff_midgray.abs().mean() < diff_black.abs().mean():
    print("\n=> Mid-gray (0.0) matches official better!")
    print("   This means our _compute_zero_vae is WRONG — it encodes -1.0 instead of 0.0")
else:
    print("\n=> Black (-1.0) matches official better!")
    print("   This means our _compute_zero_vae is correct")
