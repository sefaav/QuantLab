# QuantLab developer commands.
# Usage: make <target>

.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install install-dev format lint type-check test coverage \
        backtest walk-forward dashboard clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (runtime only)
	$(PYTHON) -m pip install -e .

install-dev: ## Install with all dev/extra dependencies
	$(PYTHON) -m pip install -e ".[dev,dashboard,yahoo,extra,docs,notebooks]"

format: ## Auto-format the codebase (Ruff formatter)
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

lint: ## Lint without modifying files
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

type-check: ## Static type checking with mypy
	$(PYTHON) -m mypy src tests scripts

test: ## Run the test suite (offline; network tests deselected)
	$(PYTHON) -m pytest -m "not network"

coverage: ## Run tests with coverage report
	$(PYTHON) -m pytest -m "not network" \
		--cov=quantlab --cov-report=term-missing --cov-report=html

backtest: ## Run the example momentum backtest
	$(PYTHON) -m quantlab.cli backtest --config configs/momentum_sp500.yaml

walk-forward: ## Run walk-forward validation for the example experiment
	$(PYTHON) -m quantlab.cli walk-forward --config configs/momentum_sp500.yaml

dashboard: ## Launch the Streamlit dashboard
	$(PYTHON) -m streamlit run src/quantlab/dashboard/app.py

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
