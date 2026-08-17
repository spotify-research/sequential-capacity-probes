"""Validate frozen assumptions inherited from the pinned eSASRec checkout."""

from __future__ import annotations

import ast
from pathlib import Path


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _constants(module: ast.Module) -> dict[str, object]:
    values = {}
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(value)
                except (ValueError, TypeError):
                    pass
    return values


def _default(module: ast.Module, function: str, argument: str) -> object:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function:
            positional = node.args.args[-len(node.args.defaults) :]
            defaults = dict(zip((item.arg for item in positional), node.args.defaults))
            return ast.literal_eval(defaults[argument])
    raise RuntimeError(f"Upstream function {function} has no default for {argument}")


def _trainer_epochs(module: ast.Module, function: str, constants: dict) -> int:
    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name != function:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            name = getattr(child.func, "id", None)
            if name != "Trainer":
                continue
            keyword = next(
                (item for item in child.keywords if item.arg == "max_epochs"), None
            )
            if keyword is None:
                break
            if isinstance(keyword.value, ast.Name):
                return int(constants[keyword.value.id])
            return int(ast.literal_eval(keyword.value))
    raise RuntimeError(f"Could not resolve max_epochs for upstream {function}")


def validate_upstream_contract(
    checkout: Path,
    generated: dict,
    dataset: str,
    protocol: dict,
) -> dict:
    """Fail before training if local claims and pinned upstream code disagree."""
    model_spec = generated["models"][0]
    fixed = model_spec["fixed_parameters"]
    if fixed.get("deterministic") is not True:
        raise RuntimeError("Upstream neural job is not deterministic")
    if not fixed.get("get_val_mask_func", "").endswith("get_val_mask_func_all"):
        raise RuntimeError("Upstream neural job does not validate on all inner users")
    for parameter in model_spec["search_parameters"]:
        if len(parameter["choices"]) != 1:
            raise RuntimeError(f"Unfrozen upstream choice: {parameter['name']}")

    trainer_module = _module(checkout / "src/models/transformers/trainer.py")
    trainer_constants = _constants(trainer_module)
    utility_module = _module(checkout / "src/utils.py")
    seed = int(_default(utility_module, "setup_deterministic", "random_seed"))
    recall_k = int(trainer_constants["RECALL_K"])
    patience = int(trainer_constants["PATIENCE"])
    trainer_name = fixed["get_trainer_func"].rsplit(".", 1)[-1]
    max_epochs = _trainer_epochs(trainer_module, trainer_name, trainer_constants)

    defaults = protocol["neural_defaults"]
    expected_epochs = (
        defaults["amazon_max_epochs"]
        if dataset in {"beauty", "sports", "toys"}
        else defaults["movielens_max_epochs"]
    )
    expected = {
        "seed": int(protocol["neural_seed"]),
        "recall_k": 10,
        "early_stopping_patience": int(defaults["early_stopping_patience"]),
        "max_epochs": int(expected_epochs),
    }
    actual = {
        "seed": seed,
        "recall_k": recall_k,
        "early_stopping_patience": patience,
        "max_epochs": max_epochs,
    }
    if actual != expected:
        raise RuntimeError(f"Pinned upstream training contract changed: {actual} != {expected}")
    if defaults["validation_metric"].lower() != f"recall@{recall_k}":
        raise RuntimeError("Configured validation metric disagrees with upstream trainer")
    return {
        **actual,
        "validation_metric": defaults["validation_metric"],
        "trainer": fixed["get_trainer_func"],
        "validation_users": fixed["get_val_mask_func"],
        "deterministic": True,
    }
