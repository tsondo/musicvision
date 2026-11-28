"""Test-session hermeticity.

``musicvision.api.app`` and the CLI call ``load_dotenv()`` at import time,
which would pull the developer's real ``.env`` (FLUX strategy override, LLM
endpoint, HF token) into the test process. Results then depended on test
order and on the machine, and tests could reach the network. Disable it before
anything imports the app, and give every test a clean, network-free env.
"""

from __future__ import annotations

import dotenv
import pytest

dotenv.load_dotenv = lambda *args, **kwargs: False  # must run before musicvision.api.app is imported

_ENV_VARS = (
    "MUSICVISION_FLUX_STRATEGY",
    "MUSICVISION_WEIGHTS_DIR",
    "LLM_BACKEND",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_DISABLE_THINKING",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "HUGGINGFACE_TOKEN",
    "HF_TOKEN",
    "SEEDVR2_REPO_DIR",
    "SEEDVR2_VENV_PYTHON",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path_factory):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Never read real weights from ~/.cache: tests that need weights set this themselves.
    monkeypatch.setenv("MUSICVISION_WEIGHTS_DIR", str(tmp_path_factory.getbasetemp() / "no_weights"))
