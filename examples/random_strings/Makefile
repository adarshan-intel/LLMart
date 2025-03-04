# Makefile for random strings example

PYTHON=python3.11
PIP=pip
VENV_DIR=venv

STRING=$(shell tr -dc 'A-Za-z0-9!#%&'\''()*+,-./:;<=>?@[\]^_{|}~' </dev/urandom | head -c 10; echo)
SCRIPT=whitebox.py
MAX_STEPS=10
LR=0.005

ARGS=--max_steps $(MAX_STEPS)  --lr $(LR)

all: check_tools install run

check_tools:
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo >&2 "$(PYTHON) is not installed. Aborting."; exit 1; }
	@command -v $(PIP) >/dev/null 2>&1 || { echo >&2 "$(PIP) is not installed. Aborting."; exit 1; }

install:
	@if [ ! -d $(VENV_DIR) ]; then \
			$(PYTHON) -m venv $(VENV_DIR); \
	fi
	. $(VENV_DIR)/bin/activate && \
	cd ../.. && \
	$(PIP) install -e ".[core,dev]";


run:
	$(VENV_DIR)/bin/$(PYTHON) $(SCRIPT) $(ARGS) "$(STRING)";

clean:
	rm -rf __pycache__ $(VENV_DIR)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Check tools, install requirements and run the script"
	@echo "  check_tools- Check if required tools are installed"
	@echo "  install    - Create virtual environment and install requirements"
	@echo "  run        - Run the script"
	@echo "  clean      - Remove temporary files and virtual environment"
	@echo "  help       - Display this help message"

.PHONY: all check_tools install run clean help