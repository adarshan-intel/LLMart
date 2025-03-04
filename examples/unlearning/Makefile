# Makefile for unlearning example

# make run MODE=gpu prompt="Who is Harry Potter?" completion"Harry Potter is the main protagonist in J.K. Rowling's series of fantasy novels"

PYTHON=python3.11
PIP=pip3
VENV_DIR=venv

max_steps=5000
max_optim_tokens=32
lr=0.0005
NUM_GPU=4
SCRIPT=whitebox.py
prompt=Who is Harry Potter?
completion=Harry Potter is the main protagonist in J.K. Rowling's series of fantasy novels
ARGS=--max_steps $(max_steps) --max_optim_tokens $(max_optim_tokens) --lr $(lr)
MODE=cpu # MODE can be either 'gpu' or 'cpu'

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
	cd ../.. && \
	$(PIP) install -e".[core,dev]"

run:
	@if [ "$(findstring use_hard_tokens, $(MAKEFLAGS))" ]; then \
		echo "Using hard tokens"; \
		ARGS="$(ARGS) --use_hard_tokens"; \
	fi; \
	if [ $(MODE) = "gpu" ]; then \
			echo "Running with $(NUM_GPU) GPUs"; \
			ts -nfG$(NUM_GPU) $(VENV_DIR)/bin/$(PYTHON) $(SCRIPT) "$(prompt)" "$(completion)" $(ARGS); \
	elif [ $(MODE) = "cpu" ]; then \
			echo "Running with CPU"; \
			$(VENV_DIR)/bin/$(PYTHON) $(SCRIPT) "$(prompt)" "$(completion)" $(ARGS); \
	else \
			echo "Invalid mode. Aborting."; exit 1; \
	fi

clean:
	rm -rf __pycache__ .installed $(VENV_DIR)

help:
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Check tools, install requirements and run the script"
	@echo "  check_tools- Check if required tools are installed"
	@echo "  install    - Create virtual environment and install requirements"
	@echo "  run        - Run the script"
	@echo "  clean      - Remove temporary files and virtual environment"
	@echo "  help       - Display this help message"
	@echo ""
	@echo "Options:"
	@echo "  prompt             - prompt provided to the model."
	@echo "  completion         - Target completion."
	@echo "  use_hard_tokens    - Enable it to use hard tokens"
	@echo "  MODE               - gpu / cpu"
	@echo "  lr                 - Learning rate for adversarial optimisation."
	@echo "  max_steps          - Optimize prompt for no more than N steps."
	@echo "  max_optim_tokens   - If optimisation fails, increase prompt size until N tokens"

.PHONY: all check_tools install run clean help