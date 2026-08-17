from __future__ import annotations

import json
from pathlib import Path

import pytest

from capacity_probes.config import load_table_config
from capacity_probes.reporting import verify_table
from tests.result_factory import ROOT, result_payload


def _write_pctm_result(output: Path, payload: dict) -> None:
    path = output / "beauty" / "pctm.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))


def test_verifier_rejects_changed_experiment_source_digest(tmp_path: Path) -> None:
    config = load_table_config(ROOT / "configs/table1.json")
    value = config["reported_ndcg10"]["beauty"]["pctm"]
    payload = result_payload(config, "beauty", "pctm", value)
    payload["environment"]["source_tree_sha256"] = "0" * 64
    _write_pctm_result(tmp_path, payload)
    with pytest.raises(ValueError, match="wrong source-tree digest"):
        verify_table(config, tmp_path, ("beauty",), ("pctm",))


def test_verifier_rejects_dirty_source_artifact(tmp_path: Path) -> None:
    config = load_table_config(ROOT / "configs/table1.json")
    value = config["reported_ndcg10"]["beauty"]["pctm"]
    payload = result_payload(config, "beauty", "pctm", value)
    payload["environment"]["source_dirty"] = True
    _write_pctm_result(tmp_path, payload)
    with pytest.raises(ValueError, match="source was dirty"):
        verify_table(config, tmp_path, ("beauty",), ("pctm",))
