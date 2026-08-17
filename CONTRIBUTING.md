# Contributing

Thank you for considering a contribution to Sequential Capacity Probes. This
repository is research software accompanying a published experimental protocol,
so reproducibility and clear provenance take priority over expanding scope.

## Before opening a change

- Use an issue to describe bugs, documentation gaps, or proposed changes.
- Do not commit datasets, model checkpoints, generated results, credentials, or
  third-party source trees.
- Keep the Table 1 protocol and frozen hyperparameters unchanged unless the
  contribution explicitly documents a new experiment rather than a reproduction.
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

The supported environment is Linux with Python 3.10.12.

```bash
git clone https://github.com/spotify-research/sequential-capacity-probes.git
cd sequential-capacity-probes
make setup
make dependency
make test
```

`make dependency` obtains the pinned public eSASRec dependency under the
gitignored `.external/` directory. Unit tests use synthetic fixtures and do not
require downloading the benchmark datasets.

## Pull requests

1. Fork the repository and create a focused feature branch.
2. Add or update tests for behavioural changes.
3. Run `make test` and include the result in the pull-request description.
4. Explain any effect on ranking, filtering, tie-breaking, data preparation,
   metrics, determinism, or reported values.
5. Update the relevant protocol, data, or hyperparameter documentation.

Keep changes reasonably scoped and avoid unrelated formatting or refactoring.
Maintainers may ask for an independently reproducible fixture before accepting
changes to model or evaluation logic.

## Licence

By contributing, you agree that your contribution is licensed under the
[Apache License 2.0](LICENSE).

