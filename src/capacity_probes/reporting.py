from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .io import atomic_json, git_head, sha256, source_tree_sha256
from .verification import validate_result


DATASET_ORDER = ("beauty", "sports", "toys", "ml1m", "ml20m")
MODEL_ORDER = (
    "mc",
    "fmc",
    "fmc_plus",
    "sasrec_plus",
    "esasrec",
    "seqrules",
    "pctm",
)
LABELS = {
    "mc": "MC",
    "fmc": "FMC",
    "fmc_plus": "FMC+",
    "sasrec_plus": "SAS+",
    "esasrec": "eSAS",
    "seqrules": "SeqRules",
    "pctm": "PCTM",
}


def verify_table(
    config: dict,
    output_root: Path,
    datasets: tuple[str, ...] = DATASET_ORDER,
    models: tuple[str, ...] = MODEL_ORDER,
) -> dict:
    rows, checks, mismatches, hashes = [], [], [], {}
    policy = config["verification"]
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "data" / "manifest.json").read_text())
    try:
        commit = git_head(root)
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    source_digest = source_tree_sha256(root)
    exact_models = set(policy["exact_4dp_models"])
    tolerances = policy["absolute_ndcg10_tolerance"]
    for dataset in datasets:
        row = {"dataset": dataset}
        for model in models:
            result_path = output_root / dataset / f"{model}.json"
            payload = json.loads(result_path.read_text())
            validate_result(
                payload, config, manifest, dataset, model, commit, source_digest
            )
            actual = float(payload["official_metrics"]["ndcg@10"])
            expected = float(config["reported_ndcg10"][dataset][model])
            delta = actual - expected
            if model in exact_models:
                passed = f"{actual:.4f}" == f"{expected:.4f}"
                criterion = "exact reported four-decimal value"
            elif model in tolerances:
                tolerance = tolerances[model]
                passed = abs(delta) <= float(tolerance)
                criterion = f"absolute NDCG@10 delta <= {float(tolerance):g}"
            else:
                raise ValueError(f"No verification policy for model: {model}")
            row[model] = actual
            hashes[str(result_path.relative_to(output_root))] = sha256(result_path)
            check = {
                "dataset": dataset,
                "model": model,
                "actual": actual,
                "reported": expected,
                "delta": delta,
                "criterion": criterion,
                "status": "pass" if passed else "fail",
            }
            checks.append(check)
            if not passed:
                mismatches.append(check)
        rows.append(row)
    _write_csv(output_root / "table1.csv", rows, models)
    _write_markdown(output_root / "RESULTS.md", rows, mismatches, models)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not mismatches else "fail",
        "criterion": {
            "default": "exact reported four-decimal NDCG@10 value",
            "model_absolute_tolerances": tolerances,
        },
        "checks": checks,
        "mismatches": mismatches,
        "result_sha256": hashes,
    }
    atomic_json(output_root / "verification.json", report)
    if mismatches:
        raise RuntimeError(f"{len(mismatches)} Table 1 values did not meet acceptance")
    return report


def _write_csv(path: Path, rows: list[dict], models: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("dataset", *models))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    rows: list[dict],
    mismatches: list[dict],
    models: tuple[str, ...],
) -> None:
    header = "| Data | " + " | ".join(LABELS[m] for m in models) + " |"
    divider = "|---|" + "---:|" * len(models)
    lines = ["# Table 1 results", "", header, divider]
    for row in rows:
        values = " | ".join(f"{row[model]:.4f}" for model in models)
        lines.append(f"| {row['dataset']} | {values} |")
    lines.extend(
        [
            "",
            "Status: " + ("PASS" if not mismatches else "FAIL"),
            "",
            "MC, SeqRules, and PCTM must match the reported four-decimal value. ",
            "FMC and FMC+ must be within absolute NDCG@10 tolerance 0.001; ",
            "SASRec+ and eSASRec must be within tolerance 0.005.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
