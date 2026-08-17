from __future__ import annotations

import subprocess
from pathlib import Path


UPSTREAM_URL = "https://github.com/blondered/transformer_benchmark.git"
UPSTREAM_COMMIT = "2b039927ceb4c7654131d7ac7c43ea124b49d240"


def ensure_dependency(root: Path, clone: bool = False) -> Path:
    checkout = root / ".external" / "transformer_benchmark"
    if not checkout.exists():
        if not clone:
            raise FileNotFoundError(
                f"Missing {checkout}; run `python scripts/setup_dependency.py`"
            )
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", UPSTREAM_URL, str(checkout)])
        subprocess.check_call(
            ["git", "-C", str(checkout), "checkout", "--detach", UPSTREAM_COMMIT]
        )
    head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f"eSASRec dependency is at {head}, expected {UPSTREAM_COMMIT}")
    changed = subprocess.run(
        ["git", "-C", str(checkout), "diff", "--quiet"], check=False
    ).returncode
    staged = subprocess.run(
        ["git", "-C", str(checkout), "diff", "--cached", "--quiet"], check=False
    ).returncode
    if changed or staged:
        raise RuntimeError("Tracked files in the eSASRec dependency were modified")
    return checkout
