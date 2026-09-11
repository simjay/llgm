.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
UV ?= uv
PORT ?= 8000
DOCS_BUILD_DIR ?= docs/_build/html
BENCHMARK_OUTPUT ?=
ENV_FILE ?=

.PHONY: help setup test test-data format lint docstrings check coverage docs docs-links serve-docs build clean
.PHONY: setup-colbert colbert-prepare test-colbert benchmark-colbert benchmark-node-search colbert-deploy colbert-shell
.PHONY: benchmark-prepare benchmark-longmemeval

help:
	@printf '%s\n' \
	  'LLGM development commands (run from the repository root)' \
	  '' \
	  '  make setup       Install editable core, development and documentation dependencies' \
	  '  make test        Run deterministic tests; exclude integration tests' \
	  '  make test-data   Run local pinned LongMemEval checks; never download data' \
	  '  make format      Format Python and sort imports with Ruff' \
	  '  make lint        Check formatting, import order and unused code with Ruff' \
	  '  make docstrings  Check every Python definition for a docstring' \
	  '  make check       Run lint, docstrings and deterministic tests' \
	  '  make coverage    Measure deterministic line/branch coverage and write reports' \
	  '  make docs        Build HTML strictly and audit rendered API/navigation' \
	  '  make docs-links  Audit repository document links separately from the user site' \
	  '  make serve-docs  Build and serve HTML at http://127.0.0.1:8000' \
	  '  make build       Build sdist and wheel, then validate metadata strictly' \
	  '  make clean       Remove only build/, dist/ and docs/_build/' \
	  '' \
	  '  make benchmark-prepare     Validate LongMemEval inputs; no model calls' \
	  '  make benchmark-longmemeval Execute LongMemEval generation and judging (paid)' \
	  '    Both require BENCHMARK_OUTPUT=NEW_DIRECTORY; execution accepts ENV_FILE=.env' \
	  '' \
	  '  make setup-colbert     Install the local Modal SDK without GPU dependencies' \
	  '  make colbert-prepare   Prepare pinned remote ColBERT assets (uses Modal)' \
	  '  make test-colbert      Run real ColBERT/PLAID GPU integration checks' \
	  '  make colbert-deploy    Deploy authenticated remote retrieval functions' \
	  '  make colbert-shell     Open the GPU environment for debugging' \
	  '' \
	  'Overrides: PYTHON=python3 UV=uv PORT=8000 DOCS_BUILD_DIR=docs/_build/html'

setup:
	@if [ "$(PYTHON)" = '.venv/bin/python' ] && [ ! -e .venv ]; then \
	  "$(UV)" venv .venv; \
	fi
	"$(UV)" pip install --python "$(PYTHON)" --group dev -e . -r docs/requirements.txt

setup-colbert:
	@if [ "$(PYTHON)" = '.venv/bin/python' ] && [ ! -e .venv ]; then \
	  "$(UV)" venv .venv; \
	fi
	"$(UV)" pip install --python "$(PYTHON)" -e '.[modal]'

colbert-prepare:
	"$(PYTHON)" -m modal run tools/colbert_modal.py --action prepare

test-colbert:
	"$(PYTHON)" -m modal run tools/colbert_modal.py --action test

benchmark-colbert:
	"$(PYTHON)" -m modal run tools/colbert_modal.py --action benchmark

benchmark-node-search:
	"$(PYTHON)" -m modal run tools/colbert_modal.py --action node-search-opaque

colbert-deploy:
	"$(PYTHON)" -m modal deploy tools/colbert_modal.py

colbert-shell:
	"$(PYTHON)" -m modal shell tools/colbert_modal.py::gpu_test

test:
	"$(PYTHON)" -m pytest -m 'not integration' -q

test-data:
	"$(PYTHON)" -m pytest tests/integration/test_longmemeval_local.py -q

benchmark-prepare:
	$(if $(strip $(BENCHMARK_OUTPUT)),,$(error BENCHMARK_OUTPUT is required; choose a new directory))
	"$(PYTHON)" -m llgm.evaluation.memory_benchmark --protocol experiments/longmemeval_pilot_v6.json --output "$(BENCHMARK_OUTPUT)"

benchmark-longmemeval:
	$(if $(strip $(BENCHMARK_OUTPUT)),,$(error BENCHMARK_OUTPUT is required; choose a new directory))
	"$(PYTHON)" -m llgm.evaluation.memory_benchmark --protocol experiments/longmemeval_pilot_v6.json --output "$(BENCHMARK_OUTPUT)" $(if $(strip $(ENV_FILE)),--env-file "$(ENV_FILE)",) --execute

lint:
	"$(PYTHON)" -m ruff check src tests tools examples docs/conf.py
	"$(PYTHON)" -m ruff format --check src tests tools examples docs/conf.py

format:
	"$(PYTHON)" -m ruff check --select I --fix src tests tools examples docs/conf.py
	"$(PYTHON)" -m ruff format src tests tools examples docs/conf.py

docstrings:
	"$(PYTHON)" tools/check_docstrings.py

check: lint docstrings test

coverage:
	mkdir -p runs/coverage/unit
	"$(PYTHON)" -m coverage run -m pytest -m 'not integration' -q
	"$(PYTHON)" -m coverage report
	"$(PYTHON)" -m coverage json
	"$(PYTHON)" -m coverage xml
	"$(PYTHON)" -m coverage html

docs:
	"$(PYTHON)" -m sphinx -E -W --keep-going -b html docs "$(DOCS_BUILD_DIR)"
	"$(PYTHON)" tools/check_docs.py "$(DOCS_BUILD_DIR)"

docs-links:
	"$(PYTHON)" tools/check_docs.py --repository-links

serve-docs: docs
	"$(PYTHON)" -m http.server "$(PORT)" --bind 127.0.0.1 --directory "$(DOCS_BUILD_DIR)"

build:
	"$(PYTHON)" -m build --installer uv
	"$(PYTHON)" -m twine check --strict dist/*.whl dist/*.tar.gz

clean:
	rm -rf build dist docs/_build
