# Makefile for autogcg example
 
# make run subset=5
 
PYTHON=python3.11
PIP=pip3
VENV_DIR=venv
 
subset=2
NUM_GPU=4
SCRIPT=main.py
ARGS=--subset $(subset)
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
    @if [ $(MODE) = "gpu" ]; then \
            echo "Running with $(NUM_GPU) GPUs"; \
            ts -nfG$(NUM_GPU) $(VENV_DIR)/bin/$(PYTHON) $(SCRIPT) $(ARGS); \
    elif [ $(MODE) = "cpu" ]; then \
            echo "Running with CPU"; \
            $(VENV_DIR)/bin/$(PYTHON) $(SCRIPT) $(ARGS); \
    else \
            echo "Invalid mode. Aborting."; exit 1; \
    fi
 
clean:
    rm -rf __pycache__ .installed $(VENV_DIR)
 
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