.PHONY: help install test lint format typecheck

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install    Install package in editable mode"
	@echo "  test       Run test suite"
	@echo "  lint       Lint and auto-fix with ruff"
	@echo "  format     Format with ruff"
	@echo "  typecheck  Type-check core library with mypy"

# On Lightning AI Studio, install into the conda base environment.
# Locally, activate a venv or conda env first — this guard prevents polluting system Python.
install:
	@if [ -z "$$VIRTUAL_ENV" ] && [ -z "$$CONDA_DEFAULT_ENV" ]; then \
		echo "ERROR: No active venv or conda environment detected. Activate one before running make install."; \
		exit 1; \
	fi
	pip install --upgrade pip setuptools wheel
	pip install -e .

test:
	python3 -m pytest tests

lint:
	ruff check --fix .

format:
	ruff format .

typecheck:
	mypy rl_eng
