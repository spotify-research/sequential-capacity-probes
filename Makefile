PYTHON ?= python3.10
VENV := .venv
RUN := $(VENV)/bin/python
PIP_VERSION := 26.2.1
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: setup dependency data data-public verify-data test run run-deterministic \
	run-factorized run-transformers run-sasrec run-esasrec verify clean-results

setup:
	$(PYTHON) -m venv $(VENV)
	$(RUN) -m pip install --upgrade pip==$(PIP_VERSION)
	$(RUN) -m pip install --requirement requirements-lock.txt
	$(RUN) -m pip install --no-deps --editable .

dependency:
	$(RUN) scripts/setup_dependency.py

data: dependency
	$(RUN) scripts/prepare_data.py --source release

data-public: dependency
	$(RUN) scripts/prepare_data.py --source public

verify-data:
	$(RUN) scripts/prepare_data.py --source verify

test:
	$(RUN) -m pytest

run:
	mkdir -p logs
	PCTM_EVAL_JOBS=$${PCTM_EVAL_JOBS:-8} $(RUN) -u scripts/run_all.py 2>&1 | tee logs/table1.log

run-deterministic:
	mkdir -p logs
	PCTM_EVAL_JOBS=$${PCTM_EVAL_JOBS:-8} $(RUN) -u scripts/run_deterministic.py 2>&1 | tee logs/deterministic.log

run-factorized:
	mkdir -p logs
	$(RUN) -u scripts/run_factorized.py 2>&1 | tee logs/factorized.log

run-transformers:
	mkdir -p logs
	$(RUN) -u scripts/run_transformers.py 2>&1 | tee logs/transformers.log

run-sasrec:
	mkdir -p logs
	$(RUN) -u scripts/run_sasrec.py 2>&1 | tee logs/sasrec.log

run-esasrec:
	mkdir -p logs
	$(RUN) -u scripts/run_esasrec.py 2>&1 | tee logs/esasrec.log

verify:
	$(RUN) scripts/verify_results.py

clean-results:
	@echo "Remove results/, cache/, and logs/ manually to start a clean run."
