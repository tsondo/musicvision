"""
WanT5Encoder — UMT5-XXL text encoder for HuMo video generation.

Wraps the HuggingFace T5EncoderModel with Wan-AI checkpoint loading.
The Wan-AI checkpoint uses a custom .pth format whose keys may differ
from the HuggingFace naming scheme.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# UMT5-XXL config that matches Wan-AI weights
_UMT5_XXL_CONFIG = dict(
    d_model=4096,
    d_ff=10240,
    num_heads=64,
    d_kv=64,
    num_layers=24,
    vocab_size=256384,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    dropout_rate=0.0,
    dense_act_fn="gelu_new",
    is_gated_act=True,
    feed_forward_proj="gated-gelu",
)


def _remap_wan_t5_keys(state_dict: dict) -> dict:
    """
    Remap Wan-AI T5 checkpoint keys to HuggingFace T5EncoderModel key names.

    The Wan-AI checkpoint stores the text encoder under various possible
    prefixes.  HuggingFace T5EncoderModel expects keys rooted at "encoder.".

    Strategy:
      1. If keys already start with "encoder.", return as-is.
      2. Strip known leading prefixes ("text_model.", "model.", "t5.").
      3. Map "layer.N." → "block.N." as used in the HF encoder stack.
      4. Fall through unchanged — strict=False in load_state_dict handles
         any residual mismatches gracefully.

    Args:
        state_dict: Raw state dict loaded from the .pth checkpoint.

    Returns:
        Remapped state dict.
    """
    # Sample a few keys to determine current naming convention
    sample_keys = list(state_dict.keys())[:8]
    log.debug("T5 checkpoint sample keys: %s", sample_keys)

    # If already in HF format, nothing to do
    if any(k.startswith("encoder.") for k in sample_keys):
        log.debug("T5 keys already in HuggingFace format — no remapping needed")
        return state_dict

    # Known prefixes used by Wan-AI / ComfyUI checkpoints
    strip_prefixes = ("text_model.", "model.", "t5.", "transformer.")

    new_sd: dict = {}
    for key, val in state_dict.items():
        new_key = key

        # Strip leading wrapper prefix
        for prefix in strip_prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break

        # HF T5EncoderModel uses "block.N" for transformer layers
        # Some checkpoints use "layer.N" instead
        if new_key.startswith("layer."):
            new_key = "block." + new_key[len("layer."):]

        # Some checkpoints omit the "encoder." root that HF expects
        if not new_key.startswith("encoder.") and not new_key.startswith("shared."):
            new_key = "encoder." + new_key

        new_sd[new_key] = val

    if new_sd.keys() != state_dict.keys():
        log.info(
            "T5 key remapping: %d keys remapped (sample before→after: %s → %s)",
            len(new_sd),
            sample_keys[0] if sample_keys else "<empty>",
            list(new_sd.keys())[0] if new_sd else "<empty>",
        )
    else:
        log.debug("T5 key remapping: no keys changed")

    return new_sd


class WanT5Encoder:
    """
    UMT5-XXL text encoder for Wan-AI / HuMo video generation.

    Usage::

        encoder = WanT5Encoder(device=torch.device("cuda:1"))
        encoder.load(Path("~/.cache/musicvision/weights/humo/shared/t5/umt5_xxl.pth"))
        pos_emb, neg_emb = encoder.encode_pair("a musician playing guitar", "")
        # pos_emb: [1, 512, 4096]
    """

    def __init__(self, device, dtype=None) -> None:
        """
        Args:
            device: Target torch.device (or str) for the model.
            dtype:  Weight dtype.  Defaults to torch.bfloat16.
        """
        import torch  # lazy — follows project convention

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.tokenizer = None
        self.model = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, weights_path: Path) -> None:
        """
        Load tokenizer and model weights from a Wan-AI .pth checkpoint.

        The tokenizer is pulled from HuggingFace (google/umt5-xxl); only the
        model weights come from the local checkpoint.  We use strict=False so
        that any key mismatches after remapping degrade gracefully rather than
        raising an error.

        Args:
            weights_path: Path to the Wan-AI T5 .pth weight file.
        """
        import torch
        from transformers import AutoTokenizer, T5Config, T5EncoderModel

        log.info("Loading WanT5Encoder tokenizer from google/umt5-xxl …")
        self.tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

        log.info("Building UMT5-XXL model (d_model=4096, num_layers=24) …")
        config = T5Config(**_UMT5_XXL_CONFIG)
        self.model = T5EncoderModel(config)

        log.info("Loading T5 weights from %s …", weights_path)
        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=False,  # .pth may contain non-tensor objects
        )
        # Unwrap common checkpoint wrappers
        if isinstance(state_dict, dict):
            for wrapper_key in ("state_dict", "model_state_dict", "model"):
                if wrapper_key in state_dict:
                    state_dict = state_dict[wrapper_key]
                    log.debug("Unwrapped checkpoint key: %s", wrapper_key)
                    break

        state_dict = _remap_wan_t5_keys(state_dict)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning("T5 load — %d missing keys (first 5: %s)", len(missing), missing[:5])
        if unexpected:
            log.warning(
                "T5 load — %d unexpected keys (first 5: %s)", len(unexpected), unexpected[:5]
            )

        log.info("Moving T5 model to %s (dtype=%s) …", self.device, self.dtype)
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        log.info("WanT5Encoder ready.")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, prompt: str, max_length: int = 512) -> "torch.Tensor":
        """
        Encode a text prompt into T5 hidden states.

        Args:
            prompt:     Input text string.
            max_length: Token sequence length (padded/truncated).  512 matches
                        the Wan-AI training setup.

        Returns:
            Tensor of shape [1, max_length, 4096] on self.device.
        """
        import torch

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("WanT5Encoder.load() must be called before encode().")

        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # last_hidden_state: [1, max_length, 4096]
        return outputs.last_hidden_state

    def encode_pair(
        self, positive: str, negative: str = ""
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Encode a positive + negative prompt pair for CFG.

        Args:
            positive: Positive conditioning text.
            negative: Negative conditioning text (empty string for unconditional).

        Returns:
            Tuple of (positive_embeds, negative_embeds), each [1, 512, 4096].
        """
        pos_emb = self.encode(positive)
        neg_emb = self.encode(negative)
        return pos_emb, neg_emb

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release model from VRAM."""
        import gc
        import torch

        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("WanT5Encoder unloaded.")
