"""
Unified LLM client supporting Anthropic Claude and OpenAI-compatible backends (vLLM).

Backend selection via environment variables:

  LLM_BACKEND = "anthropic"  (default) | "openai"

Anthropic backend:
  ANTHROPIC_API_KEY  = sk-ant-...
  ANTHROPIC_MODEL    = claude-sonnet-4-20250514  (optional, has default)

OpenAI-compatible backend (vLLM on local LAN):
  OPENAI_BASE_URL    = http://192.168.x.x:8000/v1
  OPENAI_API_KEY     = vllm          (any non-empty string for local servers)
  OPENAI_MODEL       = <model_id>    (required)

---------------------------------------------------------------------------
Current LAN vLLM server: RTX 3090 Ti 24GB (single GPU) running
cyankiwi/Qwen3.6-27B-AWQ-INT4 (~17.7GB, 4-bit AWQ, vision+text),
served under the alias `qwen` — see CLAUDE.md for the full launch command.
---------------------------------------------------------------------------
Example .env for local vLLM:
  LLM_BACKEND=openai
  OPENAI_BASE_URL=http://192.168.68.53:8000/v1
  OPENAI_API_KEY=vllm
  OPENAI_MODEL=qwen
---------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-20250514"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


@dataclass
class LLMConfig:
    """Configuration for an LLM backend connection."""

    backend: str = "anthropic"   # "anthropic" | "openai"
    model: str = ""              # model ID; falls back to env var or default
    base_url: str = ""           # vLLM endpoint, e.g. http://192.168.68.53:8000/v1
    api_key: str = ""            # explicit key; falls back to env vars
    max_tokens: int = 0  # 0 = auto (4096 for Anthropic, query vLLM server limit)
    # Reasoning models (Qwen3+) emit thinking tokens before the answer. Pipeline
    # calls need deterministic JSON, not chain-of-thought, so thinking is disabled
    # per-request by default. Set OPENAI_DISABLE_THINKING=0 to leave it on.
    disable_thinking: bool = True


def _config_from_env() -> LLMConfig:
    """Build an LLMConfig from environment variables."""
    backend = os.environ.get("LLM_BACKEND", "anthropic").lower()
    if backend == "openai":
        return LLMConfig(
            backend="openai",
            model=os.environ.get("OPENAI_MODEL", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", ""),
            api_key=os.environ.get("OPENAI_API_KEY", "vllm"),
            disable_thinking=os.environ.get("OPENAI_DISABLE_THINKING", "1") != "0",
        )
    return LLMConfig(
        backend="anthropic",
        model=os.environ.get("ANTHROPIC_MODEL", ""),
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )


class LLMClient:
    """
    Thin wrapper that routes chat completions to Anthropic or an
    OpenAI-compatible backend (e.g. vLLM).
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or _config_from_env()

    def chat(self, system: str, user: str) -> str:
        """
        Send a (system, user) message pair and return the response text.

        Raises:
            ValueError: if required config/env vars are missing.
            RuntimeError: if the API call fails.
        """
        if self.config.backend == "anthropic":
            return self._chat_anthropic(system, user)
        if self.config.backend == "openai":
            return self._chat_openai(system, user)
        raise ValueError(
            f"Unknown LLM backend: {self.config.backend!r}. "
            "Set LLM_BACKEND to 'anthropic' or 'openai'."
        )

    # ------------------------------------------------------------------
    # Private backend implementations
    # ------------------------------------------------------------------

    def _chat_anthropic(self, system: str, user: str) -> str:
        import anthropic

        key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "Anthropic API key not found. "
                "Set ANTHROPIC_API_KEY or pass api_key in LLMConfig."
            )
        model = self.config.model or ANTHROPIC_DEFAULT_MODEL
        max_tokens = self.config.max_tokens or 4096
        log.info("LLM → Anthropic (model=%s)", model)

        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()

    def _chat_openai(self, system: str, user: str) -> str:
        from openai import OpenAI

        base_url = self.config.base_url or os.environ.get("OPENAI_BASE_URL", "")
        if not base_url:
            raise ValueError(
                "OPENAI_BASE_URL not set. "
                "Example: http://192.168.68.53:8000/v1"
            )
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY", "vllm")
        model = self.config.model or os.environ.get("OPENAI_MODEL", "")
        if not model:
            raise ValueError(
                "OPENAI_MODEL not set. "
                "Set it to your vLLM served model name, e.g. qwen"
            )
        log.info("LLM → vLLM (url=%s, model=%s)", base_url, model)

        client = OpenAI(api_key=api_key, base_url=base_url)
        kwargs: dict = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens
        if self.config.disable_thinking:
            # vLLM passes chat_template_kwargs through to the model's chat
            # template; Qwen3-family templates take enable_thinking.
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = message.content or ""
        # If the server's reasoning parser routed everything into the reasoning
        # field (e.g. max_tokens exhausted mid-thought), content comes back None.
        if not content.strip():
            reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
            finish = response.choices[0].finish_reason
            raise RuntimeError(
                f"LLM returned empty content (finish_reason={finish!r}). "
                + (
                    "The response was entirely reasoning tokens — disable thinking "
                    "(OPENAI_DISABLE_THINKING=1) or raise max_tokens."
                    if reasoning
                    else "Check the vLLM server logs."
                )
            )
        # A model serving without a reasoning parser can leak its chain of
        # thought into content as <think>...</think>; strip it.
        return _THINK_BLOCK_RE.sub("", content).strip()


def get_client(config: LLMConfig | None = None) -> LLMClient:
    """
    Get an LLM client from the provided config or environment variables.

    Usage:
        client = get_client()               # reads LLM_BACKEND, etc. from env
        client = get_client(LLMConfig(      # explicit openai config
            backend="openai",
            base_url="http://192.168.68.53:8000/v1",
            model="qwen",
        ))
        text = client.chat(system_prompt, user_message)
    """
    return LLMClient(config)


def llm_available(config: LLMConfig | None = None) -> bool:
    """
    Return True if the configured LLM backend has the required credentials
    present in the environment.  Does not make a network call.

    Anthropic: requires ANTHROPIC_API_KEY (or explicit config.api_key)
    OpenAI:    requires OPENAI_BASE_URL   (or explicit config.base_url)
    """
    cfg = config or _config_from_env()
    if cfg.backend == "anthropic":
        return bool(cfg.api_key or os.environ.get("ANTHROPIC_API_KEY", ""))
    if cfg.backend == "openai":
        return bool(cfg.base_url or os.environ.get("OPENAI_BASE_URL", ""))
    return False
