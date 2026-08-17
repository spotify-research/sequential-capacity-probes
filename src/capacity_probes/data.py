from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from .dependency import ensure_dependency
from .io import csv_rows, sha256


DATASETS = ("s3_beauty", "s3_sports", "s3_toys", "ml_1m", "ml_20m")
DOWNLOAD_TIMEOUT_SECONDS = 60
PUBLIC_INPUTS = {
    "s3_beauty": (
        "beauty.txt",
        "https://raw.githubusercontent.com/RUCAIBox/CIKM2020-S3Rec/"
        "2a81540ae18615d88ef88227b0c066e5b74781e5/data/Beauty.txt",
        "Beauty.txt",
    ),
    "s3_sports": (
        "sports.txt",
        "https://raw.githubusercontent.com/RUCAIBox/CIKM2020-S3Rec/"
        "2a81540ae18615d88ef88227b0c066e5b74781e5/data/Sports_and_Outdoors.txt",
        "Sports_and_Outdoors.txt",
    ),
    "s3_toys": (
        "toys.txt",
        "https://raw.githubusercontent.com/RUCAIBox/CIKM2020-S3Rec/"
        "2a81540ae18615d88ef88227b0c066e5b74781e5/data/Toys_and_Games.txt",
        "Toys_and_Games.txt",
    ),
    "ml_1m": (
        "ml-1m.zip",
        "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "ml-1m.zip",
    ),
    "ml_20m": (
        "ml-20m.zip",
        "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        "ml-20m.zip",
    ),
}


def load_manifest(root: Path) -> dict:
    return json.loads((root / "data" / "manifest.json").read_text())


def verify_processed(root: Path, data_root: Path, datasets=DATASETS) -> None:
    expected = load_manifest(root)["processed"]
    prefixes = tuple(f"{dataset}/" for dataset in datasets)
    for relative, details in expected.items():
        if not relative.startswith(prefixes):
            continue
        path = data_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing processed split: {path}")
        actual = sha256(path)
        if actual != details["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {path}: {actual}")
        rows = csv_rows(path)
        if rows != details["rows"]:
            raise RuntimeError(f"Row-count mismatch for {path}: {rows}")


def _download(url: str, destination: Path, expected: str) -> None:
    if destination.is_file() and sha256(destination) == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(
        url, timeout=DOWNLOAD_TIMEOUT_SECONDS
    ) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    actual = sha256(temporary)
    if actual != expected:
        temporary.unlink()
        raise RuntimeError(f"Checksum mismatch for {destination.name}: {actual}")
    temporary.replace(destination)


def _extract_release(archive: Path, data_root: Path, dataset: str) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for suffix in ("leave_one_out/train.csv", "leave_one_out/holdout.csv", "statistics.csv"):
            member = f"{dataset}/{suffix}"
            if member not in bundle.namelist():
                raise RuntimeError(f"Missing {member} in {archive}")
            destination = data_root / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)


def prepare_release(root: Path, data_root: Path, datasets=DATASETS) -> None:
    manifest = load_manifest(root)
    downloads = root / "data" / "downloads"
    for dataset in datasets:
        filename = f"{dataset}.zip"
        archive = downloads / filename
        expected = manifest["release"]["archives"][filename]
        if not archive.is_file():
            raise FileNotFoundError(
                f"Missing original eSASRec archive: {archive}. "
                "Obtain it manually or use --source public."
            )
        actual = sha256(archive)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {archive}: {actual}")
        _extract_release(archive, data_root, dataset)
    verify_processed(root, data_root, datasets)


def build_public(root: Path, data_root: Path, datasets=DATASETS) -> None:
    checkout = ensure_dependency(root)
    manifest = load_manifest(root)
    for dataset in datasets:
        filename, url, manifest_name = PUBLIC_INPUTS[dataset]
        raw_path = checkout / "data" / "raw" / dataset / filename
        _download(url, raw_path, manifest["public_raw"][manifest_name])
        source = checkout / "data" / dataset
        tracked_statistics = source / "statistics.csv"
        original_statistics = tracked_statistics.read_bytes()
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    str(root / "scripts" / "build_public_dataset.py"),
                    "--dependency",
                    str(checkout),
                    "--dataset",
                    dataset,
                ]
            )
            destination = data_root / dataset
            for suffix in (
                "leave_one_out/train.csv",
                "leave_one_out/holdout.csv",
                "statistics.csv",
            ):
                target = destination / suffix
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / suffix, target)
        finally:
            tracked_statistics.write_bytes(original_statistics)
    verify_processed(root, data_root, datasets)
