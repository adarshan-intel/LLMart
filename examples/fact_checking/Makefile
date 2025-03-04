# Makefile for fact_checking example

PYTHON=python3.11
PIP=pip
VENV_DIR=.venv

NUM_STEPS=2
NUM_GPU=4
ARGS=--num_steps $(NUM_STEPS)
SCRIPT=claim.py # Can be either 'claim.py' or 'document.py'

# Because the scripts will verify the optimized suffix using the reference vllm pipeline, these examples require at least two GPUs (or a single GPU with at least 80 GB of VRAM).
MODE=gpu

all: check_tools install run

check_tools:
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo >&2 "$(PYTHON) is not installed. Aborting."; exit 1; }
	@command -v $(PIP) >/dev/null 2>&1 || { echo >&2 "$(PIP) is not installed. Aborting."; exit 1; }

install:
	@if [ ! -d $(VENV_DIR) ]; then \
			$(PYTHON) -m venv $(VENV_DIR); \
	fi
	. $(VENV_DIR)/bin/activate && \
	$(PIP) install -r requirements.txt && \
	cd ../.. && $(PIP) install -e".[core,dev]" && \
	$(PIP) install nltk && \
	$(PYTHON) -c "import nltk; nltk.download('punkt')"

run:
	@if [ $(MODE) = "gpu" ]; then \
			echo "Running with $(NUM_GPU) GPUs"; \
			ts -nfG$(NUM_GPU) $(VENV_DIR)/bin/$(PYTHON) $(SCRIPT) $(ARGS); \
	elif [ $(MODE) = "cpu" ]; then \
			echo "Cannot run in CPU mode. Aborting."; exit 1; \
	else \
			echo "Invalid mode. Aborting."; exit 1; \
	fi

clean:
	rm -rf __pycache__ 

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
