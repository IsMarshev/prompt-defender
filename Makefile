PYTHON=python3
VENV=.venv
PIP=$(VENV)/bin/pip
PY=$(VENV)/bin/python

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install:
	$(PIP) install -r requirements.txt

train:
	$(PY) train.py --config config.yaml --logger wandb

train-multi-gpu:
	$(PY) train.py --config config.yaml --devices 4 --strategy ddp

