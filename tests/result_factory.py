from __future__ import annotations

import json
import subprocess
from pathlib import Path

from capacity_probes.io import git_head, source_tree_sha256


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "data/manifest.json").read_text())
LOCAL = {"mc", "fmc", "fmc_plus", "seqrules", "pctm"}


def _configuration(config: dict, dataset: str, model: str) -> dict:
    if model == "mc":
        return {"transition_distance": 1, "smoothing": None}
    selected = config["datasets"][dataset][model]
    if model in LOCAL:
        return selected
    return {"model": selected, "training": config["protocol"]["neural_defaults"]}


def _data(config: dict, dataset: str) -> dict:
    directory = config["datasets"][dataset]["directory"]
    return {
        split: {
            **MANIFEST["processed"][f"{directory}/leave_one_out/{split}.csv"],
            "path": f"/publication-data/{directory}/leave_one_out/{split}.csv",
        }
        for split in ("train", "holdout")
    }


def _neural_params(config: dict, dataset: str, model: str) -> dict:
    selected = config["datasets"][dataset][model]
    defaults = config["protocol"]["neural_defaults"]
    ligr = model == "esasrec"
    params = {
        "n_factors": selected["factors"],
        "session_max_len": selected["max_length"],
        "n_blocks": selected["blocks"],
        "n_heads": selected["heads"],
        "dropout_rate": selected["dropout"],
        "loss": selected["loss"],
        "n_negatives": selected["negatives"],
        "lr": defaults["learning_rate"],
        "batch_size": defaults["batch_size"],
        "deterministic": True,
        "use_causal_attn": True,
        "use_key_padding_mask": ligr,
        "dataloader_num_workers": 0,
        "train_min_user_interactions": 2,
        "pos_encoding_type": "rectools.LearnableInversePositionalEncoding",
        "negative_sampler_type": "rectools.CatalogUniformSampler",
        "transformer_layers_type": (
            "src.LiGRLayers" if ligr else "rectools.SASRecTransformerLayers"
        ),
    }
    if "ff_multiplier" in selected:
        params["transformer_layers_kwargs.ff_factors_multiplier"] = selected[
            "ff_multiplier"
        ]
    return params


def result_payload(config: dict, dataset: str, model: str, ndcg: float) -> dict:
    recall = 0.2
    environment = config["protocol"]["environment"]
    try:
        source_commit, source_dirty = git_head(ROOT), False
    except (FileNotFoundError, subprocess.CalledProcessError):
        source_commit, source_dirty = None, None
    payload = {
        "model": model,
        "dataset": dataset,
        "config": _configuration(config, dataset, model),
        "official_metrics": {"ndcg@10": ndcg, "recall@10": recall},
        "data": _data(config, dataset),
        "environment": {
            "source_commit": source_commit,
            "source_dirty": source_dirty,
            "source_tree_sha256": source_tree_sha256(ROOT),
            "python": environment["python"] + " (fixture)",
            **{
                name: environment[name]
                for name in ("numpy", "pandas", "scipy", "torch", "rectools")
            },
        },
        "protocol": {"full_catalogue": True, "filter_viewed": True, "k": 10},
    }
    if model in LOCAL:
        payload["custom_metrics"] = {
            "ndcg10": ndcg,
            "recall10": recall,
            "hit_rate10": recall,
            "precision10": recall / 10,
            "reachable": 1.0,
            "users": 10,
        }
        tie_key = "factorized" if model in {"fmc", "fmc_plus"} else model
        payload["protocol"]["tie_break"] = config["protocol"][f"{tie_key}_tie_break"]
        return payload
    defaults = config["protocol"]["neural_defaults"]
    payload["model_params"] = _neural_params(config, dataset, model)
    payload["upstream"] = {"commit": config["protocol"]["esasrec_commit"]}
    payload["protocol"].update(
        {
            "seed": config["protocol"]["neural_seed"],
            "validation_metric": defaults["validation_metric"],
            "early_stopping_patience": defaults["early_stopping_patience"],
            "max_epochs": defaults[
                "amazon_max_epochs"
                if dataset in {"beauty", "sports", "toys"}
                else "movielens_max_epochs"
            ],
            "deterministic": True,
        }
    )
    return payload
