#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# type: ignore

# This Makefile is used to install dependencies for LLMart and run the tests required for release and testing.

ENV_NAME = .venv
PKG_MGR = uv

.PHONY: all
all: run

.PHONY: create-env
create-env:
	@if [ ! -d "$(ENV_NAME)" ]; then \
		echo "Creating virtual environment..."; \
		$(PKG_MGR) venv $(ENV_NAME); \
	else \
		echo "Virtual environment already exists."; \
	fi

# Install from pyproject.toml
.PHONY: install-dev
install-dev: create-env
	$(PKG_MGR) pip install -e ".[core,dev]"

# Todo: Add variables instead of direct values
# Todo: Add ARGS for cpu
.PHONY: run
run: install-dev
	. $(ENV_NAME)/bin/activate && accelerate launch -m llmart model=llama3-8b-instruct data=basic loss=model steps=1 per_device_bs=1000

.PHONY: clean
clean:
	rm -rf $(ENV_NAME)