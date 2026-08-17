from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "table1-clean-room"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_clean_room_evidence_is_complete_and_verified() -> None:
    metadata = sorted((EVIDENCE / "model-metadata").glob("*/*.json"))
    assert len(metadata) == 35

    verification = json.loads((EVIDENCE / "verification.json").read_text())
    assert verification["status"] == "pass"
    assert verification["mismatches"] == []
    assert len(verification["result_sha256"]) == 35
    for relative, expected in verification["result_sha256"].items():
        assert _sha256(EVIDENCE / "model-metadata" / relative) == expected

    with (EVIDENCE / "table1.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 5
    assert all(len(row) == 8 for row in rows)


def test_clean_room_evidence_manifest() -> None:
    for entry in (EVIDENCE / "SHA256SUMS").read_text().splitlines():
        expected, relative = entry.split("  ", 1)
        assert _sha256(EVIDENCE / relative) == expected
