.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
UV ?= uv
PORT ?= 8000
DOCS_BUILD_DIR ?= docs/_build/html
BENCHMARK_OUTPUT ?=
ENV_FILE ?=
WORKSPACE ?=
CONVERSATION_ID ?= default
VIEWER_PORT ?= 8765

.PHONY: help setup check test format lint docstrings coverage
.PHONY: docs docs-links serve-docs build clean test-data
.PHONY: viewer
.PHONY: benchmark-prepare benchmark-longmemeval
.PHONY: setup-colbert colbert-prepare test-colbert colbert-deploy colbert-shell

help:
	@printf '%s\n' \
	  'LLGM commands (run from the repository root)' \
	  '' \
	  'Everyday development' \
	  '  make viewer      Open the saved graph in your browser' \
	  '  make setup       Create .venv and install editable core, test and docs dependencies' \
	  '  make check       Run lint, docstrings and deterministic tests' \
	  '  make test        Run deterministic tests without integration tests' \
	  '  make format      Format Python and sort imports' \
	  '  make lint        Check formatting, imports and unused code' \
	  '  make docstrings  Check every Python definition for a docstring' \
	  '  make coverage    Run deterministic tests with line and branch coverage' \
	  '' \
	  'Documentation and packages' \
	  '  make docs        Build and audit the user site' \
	  '  make serve-docs  Build and preview at http://127.0.0.1:8000' \
	  '  make docs-links  Check repository document links separately' \
	  '  make build       Build and validate the wheel and source archive' \
	  '  make clean       Remove build/, dist/ and docs/_build/' \
	  '' \
	  'Optional local data check' \
	  '  make test-data   Check pinned LongMemEval data already on disk' \
	  '' \
	  'Current LongMemEval pilot (explicit opt-in)' \
	  '  make benchmark-prepare     Validate current pilot inputs without model calls' \
	  '  make benchmark-longmemeval Run pilot generation and judging (paid API calls and DSPy)' \
	  '    Set BENCHMARK_OUTPUT=NEW_DIRECTORY. Execution accepts ENV_FILE=.env.' \
	  '' \
	  'ColBERT development (remote commands can incur Modal charges)' \
	  '  make setup-colbert    Install only the local Modal SDK and library extra' \
	  '  make colbert-prepare  Prepare pinned assets in Modal' \
	  '  make test-colbert     Run real GPU integration checks' \
	  '  make colbert-deploy   Deploy authenticated retrieval functions' \
	  '  make colbert-shell    Open the GPU environment for debugging' \
	  '' \
	  'Current benchmark instructions: experiments/README.md' \
	  '' \
	  'Overrides: PYTHON=python3 UV=uv PORT=8000 DOCS_BUILD_DIR=docs/_build/html'

# Everyday development

viewer: ENV_FILE = $(wildcard .env)
viewer:
	"$(PYTHON)" -m llgm.cli view $(if $(strip $(WORKSPACE)),--workspace "$(WORKSPACE)",) --conversation-id "$(CONVERSATION_ID)" --port "$(VIEWER_PORT)" $(if $(strip $(ENV_FILE)),--env-file "$(ENV_FILE)",)

setup:
	@if [ "$(PYTHON)" = '.venv/bin/python' ] && [ ! -e .venv ]; then \
	  "$(UV)" venv .venv; \
	fi
	"$(UV)" pip install --python "$(PYTHON)" --group dev -e . -r docs/requirements.txt

test:
	"$(PYTHON)" -m pytest -m 'not integration' -q

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

# Documentation and packages

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

# Optional local data

test-data:
	"$(PYTHON)" -m pytest tests/integration/test_longmemeval_local.py -q

# Current LongMemEval evaluation

benchmark-prepare:
	$(if $(strip $(BENCHMARK_OUTPUT)),,$(error BENCHMARK_OUTPUT is required. Choose a new directory))
	"$(PYTHON)" -m llgm.evaluation.memory_benchmark --protocol experiments/longmemeval_pilot_dspy.json --output "$(BENCHMARK_OUTPUT)"

benchmark-longmemeval:
	$(if $(strip $(BENCHMARK_OUTPUT)),,$(error BENCHMARK_OUTPUT is required. Choose a new directory))
	"$(PYTHON)" -m llgm.evaluation.memory_benchmark --protocol experiments/longmemeval_pilot_dspy.json --output "$(BENCHMARK_OUTPUT)" $(if $(strip $(ENV_FILE)),--env-file "$(ENV_FILE)",) --execute

# Remote ColBERT development

setup-colbert:
	@if [ "$(PYTHON)" = '.venv/bin/python' ] && [ ! -e .venv ]; then \
	  "$(UV)" venv .venv; \
	fi
	"$(UV)" pip install --python "$(PYTHON)" -e '.[modal]'

colbert-prepare:
	"$(PYTHON)" -m modal run tools/colbert_modal.py --action prepare

test-colbert:
	"$(PYTHON)" -m modal run tools/colbert_modal.py --action test

colbert-deploy:
	"$(PYTHON)" -m modal deploy tools/colbert_modal.py

colbert-shell:
	"$(PYTHON)" -m modal shell tools/colbert_modal.py::gpu_test
