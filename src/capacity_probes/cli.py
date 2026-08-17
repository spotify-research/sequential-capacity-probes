from __future__ import annotations

import argparse
from pathlib import Path

from .config import DatasetCase, load_table_config
from .core_runner import LOCAL_MODELS, run_model
from .data import verify_processed
from .reporting import DATASET_ORDER, MODEL_ORDER, verify_table


def _selection(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if value == "all":
        return allowed
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(selected) - set(allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown values: {sorted(unknown)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper's Table 1 experiments")
    parser.add_argument("--models", default="all")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--config", type=Path, default=Path("configs/table1.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_table_config(config_path)
    data_root = args.data_root.resolve()
    cache_root, output_root = args.cache_root.resolve(), args.output_root.resolve()
    models = _selection(args.models, MODEL_ORDER)
    datasets = _selection(args.datasets, DATASET_ORDER)
    if args.verify_only:
        verify_table(config, output_root, datasets, models)
        return
    directories = tuple(config["datasets"][name]["directory"] for name in datasets)
    verify_processed(root, data_root, directories)
    for dataset in datasets:
        details = config["datasets"][dataset]
        case = DatasetCase(dataset, details["directory"], details["max_history"])
        for model in models:
            output = output_root / dataset / f"{model}.json"
            if output.exists():
                if args.resume:
                    print(f"SKIP dataset={dataset} model={model}", flush=True)
                    continue
                raise FileExistsError(f"Output already exists: {output}; use --resume")
            print(f"START dataset={dataset} model={model}", flush=True)
            if model in LOCAL_MODELS:
                run_model(
                    model,
                    case,
                    details.get(model, {}),
                    data_root,
                    cache_root,
                    output_root,
                    args.device,
                    int(config["protocol"]["refit_seed"]),
                )
            else:
                from .neural import run_neural

                run_neural(
                    root,
                    dataset,
                    model,
                    details[model],
                    config["protocol"],
                    data_root,
                    output_root,
                )
            print(f"DONE dataset={dataset} model={model}", flush=True)
    verify_table(config, output_root, datasets, models)


if __name__ == "__main__":
    main()
