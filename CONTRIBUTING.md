# Contributing to MusicVision

Thank you for considering contributing to MusicVision! We welcome bug reports, feature suggestions, documentation improvements, and code contributions.

## Contributor License Agreement

By submitting a pull request, you agree to the terms of the [MusicVision Contributor License Agreement (CLA)](CLA.md). This grants the project maintainer the right to distribute your contribution under any license, including the project's PolyForm Noncommercial license and commercial licenses. Please read the CLA before submitting.

If you are contributing on behalf of your employer, ensure you have authorization. Contact tsondo@gmail.com for corporate CLA arrangements.

## Getting Started

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Run the test suite: `pytest tests/`
5. Submit a pull request

## Code Style

- Python 3.11+
- Every module must include `from __future__ import annotations` (required by Pydantic v2 `model_rebuild()`)
- Type hints on all public function signatures
- Docstrings on all public classes and functions
- Format with `ruff format`; lint with `ruff check`

## Pipeline Architecture

Keep all pipeline logic in the core modules (`intake/`, `imaging/`, `video/`, `assembly/`). The `ui/` and `api/` layers should only call into these — never put generation logic, file management, or config handling directly in UI or API code.

## Reporting Issues

- Use GitHub Issues
- Include your GPU model, VRAM, CUDA version, and `torch.__version__`
- For inference bugs, include the full traceback and your project config (with API keys redacted)

## License

Your contributions will be licensed under the same terms as the project: [PolyForm Noncommercial License 1.0.0](LICENSE) for the open-source release, with the option for the maintainer to include contributions in commercial releases per the CLA.
