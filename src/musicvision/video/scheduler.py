"""
Flow Matching scheduler for HuMo video generation.

Reference: kijai/ComfyUI-WanVideoWrapper nodes_sampler.py — FlowMatchScheduler

Background
----------
HuMo uses a Flow Matching objective (rather than DDPM score matching).  In flow
matching, the forward process linearly interpolates between data x0 and noise ε:

    z_t = (1 - σ_t) * x0 + σ_t * ε

where σ_t is a noise level that decreases from 1.0 (pure noise) to ~0 (clean).
The model predicts the *velocity field* v = dz/dt, and the Euler update is:

    z_{t+1} = z_t + v_pred * (σ_{t+1} - σ_t)

Because σ decreases over denoising steps, (σ_{t+1} - σ_t) is negative, and the
latent moves from high noise toward clean data.

Shift parameter
---------------
A shift of σ-schedule is applied to concentrate denoising steps in the region
that matters most for video quality.  With shift > 1, more steps are allocated
to the high-noise end of the schedule:

    σ_shifted = shift * σ / (1 + (shift - 1) * σ)

The default shift=5.0 is taken from the Wan2.1 / HuMo training config.

Terminal sigma
--------------
terminal_sigma = 0.003 / 1.002 ≈ 0.002994 rather than exactly 0 prevents
numerical issues at the very end of denoising.  At this level the remaining
noise contribution is imperceptible.
"""

from __future__ import annotations

import torch


class FlowMatchScheduler:
    """
    Euler Flow Matching scheduler with shift-scaled sigma schedule.

    Usage
    -----
    ::

        scheduler = FlowMatchScheduler(num_inference_steps=50, shift=5.0)
        z = torch.randn_like(latent)   # start from pure noise

        for step_idx in range(scheduler.num_steps):
            v_pred = model(z, scheduler.sigmas[step_idx])
            z = scheduler.step(v_pred, z, step_idx)

        clean_latent = z

    Attributes
    ----------
    sigmas : torch.Tensor
        [num_inference_steps + 1] decreasing sigma values from ~1.0 to terminal.
        sigmas[0] ≈ 1.0 (fully noisy), sigmas[-1] = terminal_sigma (almost clean).
    """

    def __init__(
        self,
        num_inference_steps: int = 50,
        shift: float = 5.0,
        terminal_sigma: float = 0.003 / 1.002,
    ) -> None:
        self.num_inference_steps = num_inference_steps
        self.shift               = shift
        self.terminal_sigma      = terminal_sigma
        self._compute_sigmas()

    def _compute_sigmas(self) -> None:
        """
        Compute the sigma schedule and store as self.sigmas.

        Steps:
          1. Linear base schedule from 1.0 → terminal_sigma (inclusive endpoints).
          2. Apply shift transform: σ' = shift * σ / (1 + (shift - 1) * σ)
             This warps the schedule so more steps fall in the high-noise region.

        The result is stored on CPU as a 1D float32 tensor of length
        (num_inference_steps + 1).  It is moved to the target device lazily
        in step() and add_noise() by matching the device of the input tensors.
        """
        base = torch.linspace(
            1.0, self.terminal_sigma, self.num_inference_steps + 1,
            dtype=torch.float32,
        )   # [N+1], decreasing from 1.0 to terminal_sigma

        # Apply shift warp
        sigmas = self.shift * base / (1.0 + (self.shift - 1.0) * base)

        self.sigmas: torch.Tensor = sigmas   # [N+1], on CPU

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def step(
        self,
        v_pred: torch.Tensor,
        z_t: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        """
        Euler step in the flow matching direction.

        Args:
            v_pred:   [B, C, F, H, W]  velocity field predicted by WanModel.
            z_t:      [B, C, F, H, W]  current noisy latent.
            step_idx: int              current step index (0 … num_steps-1).

        Returns:
            [B, C, F, H, W]  updated latent z_{t+1}.

        Note
        ----
        sigma_next < sigma_curr, so the step moves z toward clean data.
        """
        device = z_t.device

        sigma_curr = self.sigmas[step_idx    ].to(device)   # scalar
        sigma_next = self.sigmas[step_idx + 1].to(device)   # scalar

        # Euler integration: z_{t+1} = z_t + v * Δσ
        # Δσ = sigma_next - sigma_curr < 0  (denoising direction)
        delta_sigma = sigma_next - sigma_curr   # negative scalar

        return z_t + v_pred * delta_sigma

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        """
        Forward diffusion: mix clean latent with noise at a given sigma level.

        Used to create noisy inputs for training or for inversion.

        Args:
            clean:    [B, C, F, H, W]  clean (denoised) latent x0.
            noise:    [B, C, F, H, W]  Gaussian noise ε ~ N(0, I).
            step_idx: int              sigma index in self.sigmas.

        Returns:
            [B, C, F, H, W]  noisy latent z_t = (1 - σ) * x0 + σ * ε.
        """
        device = clean.device
        sigma = self.sigmas[step_idx].to(device)   # scalar

        return (1.0 - sigma) * clean + sigma * noise

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        """Number of denoising steps (alias for num_inference_steps)."""
        return self.num_inference_steps

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FlowMatchScheduler("
            f"num_steps={self.num_inference_steps}, "
            f"shift={self.shift:.2f}, "
            f"terminal_sigma={self.terminal_sigma:.6f}, "
            f"sigma_range=[{self.sigmas[0].item():.4f}, {self.sigmas[-1].item():.6f}]"
            f")"
        )
