PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

CONFIG ?= config.yaml
RUN_ROOT ?= runs
GRID ?= grid.example.yaml
MODEL_PATH ?= models/guard_model
DATA_PATH ?= thinking.jsonl
EXPORT_DIR ?= artifacts/guard_model
CHECKPOINT ?=
EXTRA ?=

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install:
	$(PIP) install -r requirements.txt

train:
	$(PY) train.py --config $(CONFIG) --logger tensorboard $(EXTRA)

train-multi-gpu:
	$(PY) train.py --config $(CONFIG) --devices 4 --strategy ddp $(EXTRA)

export:
	$(PY) export_checkpoint.py --config $(CONFIG) --checkpoint $(CHECKPOINT) --output-dir $(EXPORT_DIR) $(EXTRA)

eval:
	$(PY) eval.py --model-path $(MODEL_PATH) --data-path $(DATA_PATH) $(EXTRA)

experiment:
	$(PY) run_experiment.py --config $(CONFIG) --run-root $(RUN_ROOT) $(EXTRA)

grid:
	$(PY) run_grid.py --grid $(GRID) --run-root $(RUN_ROOT) $(EXTRA)
