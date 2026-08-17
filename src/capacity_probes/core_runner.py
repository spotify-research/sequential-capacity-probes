from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import DatasetCase
from .io import environment_manifest, immutable_json, read_interactions, sha256
from .metrics import official_metrics
from .sequences import EvaluationSet, SequenceStore


LOCAL_MODELS = ("mc", "fmc", "fmc_plus", "seqrules", "pctm")


def _load_case(data_root: Path, case: DatasetCase):
    directory = data_root / case.directory / "leave_one_out"
    train_path, holdout_path = directory / "train.csv", directory / "holdout.csv"
    train, holdout = read_interactions(train_path), read_interactions(holdout_path)
    store = SequenceStore.from_frame(train)
    evaluation = EvaluationSet.from_holdout(store, holdout)
    provenance = {
        "train": {"path": str(train_path), "sha256": sha256(train_path), "rows": len(train)},
        "holdout": {
            "path": str(holdout_path),
            "sha256": sha256(holdout_path),
            "rows": len(holdout),
        },
    }
    return train, holdout, store, evaluation, provenance


def _build_and_recommend(
    model_name: str,
    store: SequenceStore,
    evaluation: EvaluationSet,
    values: dict,
    cache_dir: Path,
    max_history: int,
    device_name: str,
    seed: int,
):
    if model_name == "mc":
        from .models.mc import TIE_BREAK, build_mc, recommend_mc

        model = build_mc(store, cache_dir)
        output = recommend_mc(model, evaluation)
    elif model_name == "pctm":
        from .models.pctm import TIE_BREAK, build_pctm, recommend_pctm

        model = build_pctm(store, cache_dir, values)
        output = recommend_pctm(model, evaluation, max_history)
    elif model_name == "seqrules":
        from .models.seqrules import TIE_BREAK, build_seqrules, recommend_seqrules

        model = build_seqrules(store, cache_dir, values)
        output = recommend_seqrules(model, evaluation, max_history)
    elif model_name == "fmc":
        from .models.fmc import TIE_BREAK, build_fmc, recommend_fmc

        model = build_fmc(store, max_history, values, device_name, seed)
        output = recommend_fmc(model, evaluation)
    elif model_name == "fmc_plus":
        from .models.fmc_plus import TIE_BREAK, build_fmc_plus, recommend_fmc_plus

        model = build_fmc_plus(store, max_history, values, device_name, seed)
        output = recommend_fmc_plus(model, evaluation)
    else:
        raise ValueError(f"Unknown local model: {model_name}")
    return model, *output, TIE_BREAK


def run_model(
    model_name: str,
    case: DatasetCase,
    config: dict,
    data_root: Path,
    cache_root: Path,
    output_root: Path,
    device_name: str,
    seed: int,
) -> dict:
    """Orchestrate one local model without containing model-specific logic."""
    if model_name not in LOCAL_MODELS:
        raise ValueError(f"Not a local model: {model_name}")
    train, holdout, store, evaluation, provenance = _load_case(data_root, case)
    model, custom, recommendations, tie_break = _build_and_recommend(
        model_name,
        store,
        evaluation,
        config,
        cache_root / model_name / case.name,
        case.max_history,
        device_name,
        seed,
    )
    official = official_metrics(recommendations, train, holdout)
    result = _result(
        model_name,
        case,
        asdict(model.config),
        custom,
        official,
        provenance,
        tie_break,
    )
    immutable_json(output_root / case.name / f"{model_name}.json", result)
    return result


def _result(model, case, config, custom, official, provenance, tie_break) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "dataset": case.name,
        "config": config,
        "custom_metrics": asdict(custom),
        "official_metrics": official,
        "data": provenance,
        "environment": environment_manifest(),
        "protocol": {
            "full_catalogue": True,
            "filter_viewed": True,
            "k": 10,
            "tie_break": tie_break,
        },
    }
