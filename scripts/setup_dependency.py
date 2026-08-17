#!/usr/bin/env python3
from pathlib import Path

from capacity_probes.dependency import ensure_dependency


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(ensure_dependency(root, clone=True))

