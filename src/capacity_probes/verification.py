"""Independent validation of result identity, provenance, and protocol."""

from __future__ import annotations

import math


LOCAL_MODELS = {"mc", "fmc", "fmc_plus", "seqrules", "pctm"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _expected_config(config: dict, dataset: str, model: str) -> dict:
    if model == "mc":
        return {"transition_distance": 1, "smoothing": None}
    selected = config["datasets"][dataset][model]
    if model in LOCAL_MODELS:
        return selected
    return {"model": selected, "training": config["protocol"]["neural_defaults"]}


def _validate_metrics(payload: dict, model: str) -> None:
    official = payload.get("official_metrics", {})
    for name in ("ndcg@10", "recall@10"):
        value = official.get(name)
        _require(isinstance(value, (int, float)), f"{model}: missing {name}")
        _require(math.isfinite(value) and 0 <= value <= 1, f"{model}: invalid {name}")
    if model not in LOCAL_MODELS:
        return
    custom = payload.get("custom_metrics", {})
    pairs = (("ndcg10", "ndcg@10"), ("recall10", "recall@10"))
    for custom_name, official_name in pairs:
        difference = abs(float(custom[custom_name]) - float(official[official_name]))
        _require(difference <= 1e-12, f"{model}: metric audit differs by {difference}")
    _require(int(custom.get("users", 0)) > 0, f"{model}: no evaluated users")


def _validate_data(
    payload: dict, config: dict, manifest: dict, dataset: str, model: str
) -> None:
    directory = config["datasets"][dataset]["directory"]
    for split in ("train", "holdout"):
        key = f"{directory}/leave_one_out/{split}.csv"
        expected = manifest["processed"][key]
        actual = payload.get("data", {}).get(split, {})
        _require(actual.get("sha256") == expected["sha256"], f"{model}: bad {split} hash")
        _require(int(actual.get("rows", -1)) == expected["rows"], f"{model}: bad {split} rows")


def _validate_environment(
    payload: dict,
    config: dict,
    commit: str | None,
    source_digest: str,
    model: str,
) -> None:
    environment = payload.get("environment", {})
    _require(
        environment.get("source_tree_sha256") == source_digest,
        f"{model}: wrong source-tree digest",
    )
    if commit is None:
        recorded_commit = environment.get("source_commit")
        _require(
            recorded_commit is None
            or (
                isinstance(recorded_commit, str)
                and len(recorded_commit) == 40
                and all(character in "0123456789abcdef" for character in recorded_commit)
            ),
            f"{model}: malformed recorded source commit",
        )
        _require(
            environment.get("source_dirty") in (False, None),
            f"{model}: source was dirty",
        )
    else:
        _require(environment.get("source_commit") == commit, f"{model}: wrong source commit")
        _require(environment.get("source_dirty") is False, f"{model}: source was dirty")
    expected = config["protocol"]["environment"]
    python = str(environment.get("python", "")).split()[0]
    _require(python == expected["python"], f"{model}: Python {python} is not frozen")
    for package in ("numpy", "pandas", "scipy", "torch", "rectools"):
        actual = str(environment.get(package, "")).split("+", 1)[0]
        _require(actual == expected[package], f"{model}: {package} {actual} is not frozen")


def _validate_neural(payload: dict, config: dict, dataset: str, model: str) -> None:
    selected = config["datasets"][dataset][model]
    defaults = config["protocol"]["neural_defaults"]
    params = payload.get("model_params", {})
    mapping = {
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
        "dataloader_num_workers": 0,
        "train_min_user_interactions": 2,
    }
    for name, expected in mapping.items():
        _require(params.get(name) == expected, f"{model}: resolved {name} is not frozen")
    layer = str(params.get("transformer_layers_type", ""))
    expected_layer = "LiGRLayers" if model == "esasrec" else "SASRecTransformerLayers"
    _require(layer.endswith(expected_layer), f"{model}: wrong transformer layer")
    _require(
        params.get("use_key_padding_mask") == (model == "esasrec"),
        f"{model}: wrong padding mask",
    )
    _require(
        str(params.get("pos_encoding_type", "")).endswith(
            "LearnableInversePositionalEncoding"
        ),
        f"{model}: wrong positional encoding",
    )
    _require(
        str(params.get("negative_sampler_type", "")).endswith(
            "CatalogUniformSampler"
        ),
        f"{model}: wrong negative sampler",
    )
    if "ff_multiplier" in selected:
        _require(
            params.get("transformer_layers_kwargs.ff_factors_multiplier")
            == selected["ff_multiplier"],
            f"{model}: wrong feed-forward multiplier",
        )
    upstream = payload.get("upstream", {})
    _require(
        upstream.get("commit") == config["protocol"]["esasrec_commit"],
        f"{model}: wrong upstream commit",
    )
    protocol = payload.get("protocol", {})
    expected_epochs = defaults[
        "amazon_max_epochs" if dataset in {"beauty", "sports", "toys"} else "movielens_max_epochs"
    ]
    for key, expected in {
        "seed": config["protocol"]["neural_seed"],
        "validation_metric": defaults["validation_metric"],
        "early_stopping_patience": defaults["early_stopping_patience"],
        "max_epochs": expected_epochs,
        "deterministic": True,
    }.items():
        _require(protocol.get(key) == expected, f"{model}: wrong neural {key}")


def validate_result(
    payload: dict,
    config: dict,
    manifest: dict,
    dataset: str,
    model: str,
    commit: str | None,
    source_digest: str,
) -> None:
    """Validate one result before accepting its Table 1 value."""
    _require(payload.get("dataset") == dataset, f"{model}: dataset identity mismatch")
    _require(payload.get("model") == model, f"{model}: model identity mismatch")
    _require(
        payload.get("config") == _expected_config(config, dataset, model),
        f"{model}: configuration mismatch",
    )
    _validate_metrics(payload, model)
    _validate_data(payload, config, manifest, dataset, model)
    _validate_environment(payload, config, commit, source_digest, model)
    protocol = payload.get("protocol", {})
    for key, expected in {"full_catalogue": True, "filter_viewed": True, "k": 10}.items():
        _require(protocol.get(key) == expected, f"{model}: wrong protocol {key}")
    if model in LOCAL_MODELS:
        tie_key = "factorized" if model in {"fmc", "fmc_plus"} else model
        expected_tie = config["protocol"][f"{tie_key}_tie_break"]
        _require(protocol.get("tie_break") == expected_tie, f"{model}: tie metadata mismatch")
    else:
        _validate_neural(payload, config, dataset, model)
