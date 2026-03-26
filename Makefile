PYTHON ?= python3
VENV ?= $(if $(wildcard .venv),.venv,$(if $(wildcard venv),venv,.venv))
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
CLI := prompt_defender.cli

CONFIG ?= config.yaml
GRID ?= grid.example.yaml
RUN_ROOT ?=
MODEL_PATH ?= checkpoints/hf_model
DATA_PATH ?= thinking.jsonl
EXPORT_DIR ?= artifacts/guard_model
CHECKPOINT ?=
PROMPT ?=
RESPONSE ?=
EXTRA ?=
WANDB_KEY=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

.PHONY: help setup install train train-multi-gpu export eval infer experiment grid

help:
	@printf '%s\n' \
		'Available targets:' \
		'  setup            Create virtualenv and upgrade pip' \
		'  install          Install requirements into the virtualenv' \
		'  train            Run single-GPU training' \
		'  train-multi-gpu  Run multi-GPU training with DDP' \
		'  export           Export a Lightning checkpoint to HF format' \
		'  eval             Evaluate an exported model' \
		'  infer            Run one interactive inference sample' \
		'  experiment       Run train -> export -> eval pipeline' \
		'  grid             Run a grid of experiments' \
		'' \
		'Common vars:' \
		'  VENV=.venv|venv  Virtualenv path (auto-detected by default)' \
		'  CONFIG=...       Base config file' \
		'  RUN_ROOT=...     Output root for experiment/grid runs' \
		'  EXTRA="..."      Extra CLI flags passed to the target'

$(PY):
	$(PYTHON) -m venv $(VENV)

setup: $(PY)
	$(PIP) install --upgrade pip

install: setup
	$(PIP) install -r requirements.txt
	$(PY) -m wandb login $(WANDB_KEY)

train: $(PY)
	$(PY) -m $(CLI).train --config "$(CONFIG)" --logger wandb $(EXTRA)

train-multi-gpu: $(PY)
	$(PY) -m $(CLI).train --config "$(CONFIG)" --devices 2 --strategy ddp --logger wandb $(EXTRA) 

export: $(PY)
	@test -n "$(CHECKPOINT)" || (echo "CHECKPOINT is required: make export CHECKPOINT=path/to/model.ckpt"; exit 1)
	$(PY) -m $(CLI).export_checkpoint --config "$(CONFIG)" --checkpoint "$(CHECKPOINT)" --output-dir "$(EXPORT_DIR)" $(EXTRA)

eval: $(PY)
	$(PY) -m $(CLI).eval --model-path "$(MODEL_PATH)" --data-path "$(DATA_PATH)" --batch-size 16 $(EXTRA) 

infer: $(PY)
	@test -n "$(PROMPT)" || (echo "PROMPT is required: make infer MODEL_PATH=... PROMPT='...'" ; exit 1)
	$(PY) -m $(CLI).infer_guard --model-path "$(MODEL_PATH)" --prompt "$(PROMPT)" $(if $(RESPONSE),--response "$(RESPONSE)",) $(EXTRA)

experiment: $(PY)
	$(PY) -m $(CLI).run_experiment --config "$(CONFIG)" $(if $(RUN_ROOT),--run-root "$(RUN_ROOT)",) $(EXTRA)

grid: $(PY)
	$(PY) -m $(CLI).run_grid --grid "$(GRID)" $(if $(RUN_ROOT),--run-root "$(RUN_ROOT)",) $(EXTRA)
