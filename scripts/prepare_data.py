#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from capacity_probes.data import (
    DATASETS,
    build_public,
    prepare_release,
    verify_processed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("release", "public", "verify"), default="release")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve()
    if args.source == "release":
        prepare_release(root, data_root, args.datasets)
    elif args.source == "public":
        build_public(root, data_root, args.datasets)
    else:
        verify_processed(root, data_root, args.datasets)
    print(f"Verified data at {data_root}")


if __name__ == "__main__":
    main()
