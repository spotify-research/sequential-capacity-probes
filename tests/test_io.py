from __future__ import annotations

import json
from pathlib import Path

from capacity_probes.io import immutable_json, source_tree_sha256


def test_immutable_json_creates_once_and_rejects_replacement(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    immutable_json(path, {"value": 1})
    assert json.loads(path.read_text()) == {"value": 1}
    try:
        immutable_json(path, {"value": 2})
    except FileExistsError as error:
        assert "Refusing to replace" in str(error)
    else:
        raise AssertionError("Existing result artifact was replaced")
    assert json.loads(path.read_text()) == {"value": 1}


def test_source_tree_digest_covers_experiment_files_only(tmp_path: Path) -> None:
    (tmp_path / "src/package").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "data").mkdir()
    for name in (
        "configs/table1.json",
        "data/manifest.json",
        "Makefile",
        "pyproject.toml",
        "requirements-lock.txt",
        "src/package/model.py",
        "scripts/run.py",
    ):
        (tmp_path / name).write_text(name)
    before = source_tree_sha256(tmp_path)
    (tmp_path / "README.md").write_text("documentation-only change")
    assert source_tree_sha256(tmp_path) == before
    (tmp_path / "src/package/model.py").write_text("scientific-code change")
    assert source_tree_sha256(tmp_path) != before
