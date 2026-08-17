#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
import zipfile
from pathlib import Path


def extract_zip(archive: Path, destination: Path) -> None:
    """Extract a hash-verified public ZIP without requiring a system binary."""
    root = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}") from error
            if stat.S_ISLNK(member.external_attr >> 16):
                raise RuntimeError(f"ZIP symlink is not allowed: {member.filename}")
        bundle.extractall(root)


def prepare_movielens_archive(module: object) -> None:
    raw = Path("data/raw") / module.DATASET_NAME
    interactions = raw / module.INTERACTIONS_FILENAME
    if interactions.is_file():
        return
    archive = raw / module.ZIP_FILENAME
    if not archive.is_file():
        raise FileNotFoundError(f"Missing hash-verified public archive: {archive}")
    extract_zip(archive, raw)
    extracted = raw / module.EXTRACTED_DIRNAME
    if not (extracted / module.INTERACTIONS_FILENAME).is_file():
        raise RuntimeError(f"Missing interactions file in {archive}")
    for source in extracted.iterdir():
        destination = raw / source.name
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        elif destination.exists() or destination.is_symlink():
            destination.unlink()
        shutil.move(str(source), destination)
    extracted.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("s3_beauty", "s3_sports", "s3_toys", "ml_1m", "ml_20m"),
        required=True,
    )
    args = parser.parse_args()
    dependency = args.dependency.resolve()
    os.chdir(dependency)
    sys.path.insert(0, str(dependency))
    from src.datasets.common import process_validation_schemes

    if args.dataset.startswith("s3_"):
        from src.datasets.s3_repro import process_raw_file

        variant = args.dataset.removeprefix("s3_")
        process_validation_schemes(
            args.dataset,
            lambda path: process_raw_file(path, variant),
            select_val_schemes=["leave_one_out.py"],
        )
        return
    module_name = "src.datasets.ml_1m" if args.dataset == "ml_1m" else "src.datasets.ml_20m"
    module = __import__(module_name, fromlist=["unused"])
    prepare_movielens_archive(module)
    process_validation_schemes(
        module.DATASET_NAME,
        module.process_raw_file,
        select_val_schemes=["leave_one_out.py"],
    )


if __name__ == "__main__":
    main()
