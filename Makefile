PYTHON ?= python3
PY = .venv/bin/python
INSTALLED = .venv/.installed

.PHONY: setup eda experiment neural horizons anomaly report all slides test lint
# The pipeline stages read each other's outputs, so they must never run in parallel.
.NOTPARALLEL:

# Every command below sets up the environment first if needed, so `make all` or
# `make test` work on a fresh clone; `make setup` alone is optional.
setup: $(INSTALLED)

$(INSTALLED): requirements.txt requirements-dev.txt pyproject.toml
	@$(PYTHON) -c 'import sys; sys.exit(sys.version_info < (3, 12))' || \
	  { echo "Python 3.12 or newer is needed; try: make PYTHON=python3.12 setup"; exit 1; }
	$(PYTHON) -m venv .venv
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install --no-deps -e .
	@touch $@

eda: $(INSTALLED)
	@echo "[1/6] Data checks and EDA figures (~1 min)"
	$(PY) -m ettpoc.eda
experiment: $(INSTALLED)
	@echo "[2/6] Main experiment: tuning, then 12 monthly folds x 2 stations x 2 framings (~20 min)"
	$(PY) -m ettpoc.experiment
neural: $(INSTALLED)
	@echo "[3/6] LSTM on the same folds (~5 min)"
	$(PY) -m ettpoc.neural
horizons: $(INSTALLED)
	@echo "[4/6] Horizon sweep, 1-48 h (~15 min)"
	$(PY) -m ettpoc.horizons
anomaly: $(INSTALLED)
	@echo "[5/6] Fault-detection study (~2 min)"
	$(PY) -m ettpoc.anomaly
report: $(INSTALLED)
	@echo "[6/6] Tables and figures for the report (~2 min)"
	$(PY) -m ettpoc.report

all: eda experiment neural horizons anomaly report

# Report PDFs, Japanese and English (need xelatex and the Hiragino Sans font, i.e. macOS);
# run after `make all`.
XELATEX = xelatex -interaction=nonstopmode -halt-on-error -output-directory=build

slides:
	cd slides && mkdir -p build && for deck in ETT_PoC_Report_JA ETT_PoC_Report_EN; do \
	  $(XELATEX) $$deck.tex && $(XELATEX) $$deck.tex && cp build/$$deck.pdf . || exit 1; done

test: $(INSTALLED) ; $(PY) -m pytest -q
lint: $(INSTALLED) ; .venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
