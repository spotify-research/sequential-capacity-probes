from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from rectools import Columns


SOURCE_FILES = (
    "Makefile",
    "configs/table1.json",
    "data/manifest.json",
    "pyproject.toml",
    "requirements-lock.txt",
)


def read_interactions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={Columns.User: "int64", Columns.Item: "int64"},
        parse_dates=[Columns.Datetime],
    )
    required = {Columns.User, Columns.Item, Columns.Datetime}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(required - set(frame))}")
    frame["_source_order"] = np.arange(len(frame), dtype=np.int64)
    return frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
    """Hash every file that defines an experiment, independent of Git metadata."""
    paths = [root / name for name in SOURCE_FILES]
    paths.extend((root / "src").rglob("*.py"))
    paths.extend((root / "scripts").glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda candidate: candidate.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def immutable_json(path: Path, payload: dict) -> None:
    """Atomically create a JSON artifact without replacing an existing result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to replace result artifact: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def environment_manifest() -> dict:
    import rectools

    try:
        import torch

        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch_version, torch_cuda, cuda_available = None, None, False

    root = Path(__file__).resolve().parents[2]
    try:
        source_commit = git_head(root)
        source_dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        source_commit, source_dirty = None, None
    return {
        "command": sys.argv,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "source_tree_sha256": source_tree_sha256(root),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "torch": torch_version,
        "torch_cuda": torch_cuda,
        "cuda_available": cuda_available,
        "rectools": rectools.__version__,
    }


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def csv_rows(path: Path) -> int:
    with path.open("rb") as source:
        lines = sum(block.count(b"\n") for block in iter(lambda: source.read(1 << 20), b""))
    return lines - 1
