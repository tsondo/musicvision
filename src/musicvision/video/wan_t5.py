"""
WanT5Encoder — UMT5-XXL text encoder for HuMo video generation.

Uses the vendored Wan-AI T5 architecture (vendor/wan_t5_arch.py) which
matches the Wan-AI checkpoint key format exactly:
  token_embedding, blocks.N.{norm1,attn,norm2,ffn,pos_embedding}, norm

The Wan-AI checkpoint does NOT use HuggingFace T5EncoderModel naming —
it has its own custom T5Encoder with different layer names and structure.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Default text sequence length (matches Wan-AI training config)
_DEFAULT_TEXT_LEN = 512


class WanT5Encoder:
    """
    UMT5-XXL text encoder for Wan-AI / HuMo video generation.

    Wraps the vendored Wan-AI T5EncoderModel which builds the correct
    architecture and loads checkpoints with no key remapping needed.

    Usage::

        encoder = WanT5Encoder(device=torch.device("cuda:1"))
        encoder.load(Path("~/.cache/musicvision/weights/shared/t5/models_t5_umt5-xxl-enc-bf16.pth"))
        context = encoder.encode("a musician playing guitar")
        # context: list of [L_i, 4096] tensors (variable length, trimmed by attention mask)
    """

    def __init__(self, device, dtype=None, text_len: int = _DEFAULT_TEXT_LEN) -> None:
        import torch

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.text_len = text_len
        self._model = None  # Wan-AI T5EncoderModel instance

    def load(self, weights_path: Path) -> None:
        """
        Build the Wan-AI T5Encoder, load checkpoint weights, and init tokenizer.

        The vendored T5EncoderModel handles everything: model construction,
        weight loading (strict), and tokenizer init.
        """
        from musicvision.video.vendor.wan_t5_arch import T5EncoderModel

        log.info("Loading WanT5Encoder from %s onto %s (dtype=%s) …",
                 weights_path.name, self.device, self.dtype)

        self._model = T5EncoderModel(
            text_len=self.text_len,
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=str(weights_path),
            tokenizer_path='google/umt5-xxl',
        )

        log.info("WanT5Encoder ready.")

    def _pad_context(self, context_list: list) -> "torch.Tensor":
        """
        Pad variable-length T5 outputs to fixed [1, text_len, 4096].

        The Wan-AI T5EncoderModel returns a list of [L_i, 4096] tensors
        (one per input string, trimmed to actual token count). Our DiT
        (wan_model.py, ported from ComfyUI) expects pre-padded
        [B, text_len, 4096] tensors, so we zero-pad here.
        """
        import torch

        # context_list is a list with one element per input string
        # Each element is [L_i, 4096] where L_i <= text_len
        padded = torch.stack([
            torch.cat([
                u,
                u.new_zeros(self.text_len - u.size(0), u.size(1)),
            ])
            for u in context_list
        ])  # [B, text_len, 4096]
        return padded

    def encode(self, prompt: str) -> "torch.Tensor":
        """
        Encode a text prompt into T5 hidden states.

        Args:
            prompt: Input text string.

        Returns:
            Tensor of shape [1, text_len, 4096] zero-padded to text_len.
        """
        if self._model is None:
            raise RuntimeError("WanT5Encoder.load() must be called before encode().")

        context_list = self._model([prompt], self.device)
        return self._pad_context(context_list)

    def encode_pair(
        self, positive: str, negative: str = ""
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Encode a positive + negative prompt pair for CFG.

        Args:
            positive: Positive conditioning text.
            negative: Negative conditioning text (empty string for unconditional).

        Returns:
            Tuple of (pos_embeds, neg_embeds), each [1, text_len, 4096].
        """
        pos = self.encode(positive)
        neg = self.encode(negative)
        return pos, neg

    def unload(self) -> None:
        """Release model from VRAM."""
        import gc
        import torch

        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        log.info("WanT5Encoder unloaded.")
