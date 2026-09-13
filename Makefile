# PaperPilot developer tasks.
.PHONY: help install fixture benchmark demo test lint fix cov eval serve docker clean

PY ?= .venv/bin/python
FIXTURE := fixtures/generated/synthetic_paper.pdf

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install in editable mode
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -e ".[dev]"

fixture:  ## Generate the synthetic fixture paper
	$(PY) fixtures/synthetic_paper.py $(FIXTURE)

demo: fixture  ## Analyse the fixture paper end to end
	$(PY) -m paperpilot analyze $(FIXTURE) --output reports/example.md

test:  ## Run the test suite
	$(PY) -m pytest

cov:  ## Run the test suite with coverage
	$(PY) -m pytest --cov=paperpilot --cov-report=term-missing

lint:  ## Lint
	$(PY) -m ruff check src tests fixtures scripts

fix:  ## Auto-fix lint issues
	$(PY) -m ruff check --fix src tests fixtures scripts

benchmark:  ## Build the benchmark set (PDFs are generated, not committed)
	$(PY) fixtures/synthetic_paper.py fixtures/benchmark/synthetic_paper.pdf

eval: benchmark  ## Run the evaluation harness on the benchmark set
	$(PY) scripts/evaluate.py --papers ./fixtures/benchmark --output ./eval_results

serve:  ## Run the API and web UI
	$(PY) -m paperpilot serve

docker:  ## Build and run the container
	docker compose up --build

clean:  ## Remove caches and generated artefacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov data eval_results
	rm -rf fixtures/generated reports/*.md
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
