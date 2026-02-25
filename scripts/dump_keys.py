#!/usr/bin/env python3
"""
Dump the first N keys from a model checkpoint.

Supports: .safetensors, .pt / .pth (torch), .gguf

Usage:
    python scripts/dump_keys.py path/to/weights.pth 20
    python scripts/dump_keys.py path/to/model.safetensors 50
    python scripts/dump_keys.py path/to/model.gguf 30
"""

import sys
from pathlib import Path


def dump_safetensors(path: Path, n: int) -> None:
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    print(f"Total keys: {len(keys)}")
    for k in keys[:n]:
        print(k)


def dump_torch(path: Path, n: int) -> None:
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        # unwrap common wrappers
        for wrapper in ("state_dict", "model", "module", "model_state_dict"):
            if wrapper in state and isinstance(state[wrapper], dict):
                print(f"(unwrapped '{wrapper}' key)")
                state = state[wrapper]
                break
        keys = list(state.keys())
    else:
        print(f"Warning: loaded object is {type(state)}, not a dict — printing attributes")
        keys = [str(k) for k in dir(state)]
    print(f"Total keys: {len(keys)}")
    for k in keys[:n]:
        print(k)


def dump_gguf(path: Path, n: int) -> None:
    import gguf
    reader = gguf.GGUFReader(str(path))
    keys = [t.name for t in reader.tensors]
    print(f"Total tensors: {len(keys)}")
    for k in keys[:n]:
        print(k)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: dump_keys.py <checkpoint_path> [n=20]")
        sys.exit(1)

    path = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"File:   {path}")
    print(f"Format: {path.suffix}")
    print(f"Showing first {n} keys\n" + "-" * 40)

    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        dump_safetensors(path, n)
    elif suffix in (".pt", ".pth", ".bin"):
        dump_torch(path, n)
    elif suffix == ".gguf":
        dump_gguf(path, n)
    else:
        print(f"Unknown extension '{suffix}' — trying torch first")
        dump_torch(path, n)


if __name__ == "__main__":
    main()
