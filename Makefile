PYTHON ?= python3
PY = .venv/bin/python
INSTALLED = .venv/.installed
PIP = $(PY) -m pip install -q --disable-pip-version-check

BOLD  := $(shell tput bold 2>/dev/null)
DIM   := $(shell tput dim 2>/dev/null)
GREEN := $(shell tput setaf 2 2>/dev/null)
RESET := $(shell tput sgr0 2>/dev/null)

# $(call banner,step,title,expected duration)
define banner
	@printf '\n$(BOLD)[%s] %s$(RESET)  $(DIM)%s$(RESET)\n' "$(1)" "$(2)" "$(3)"
endef

.DEFAULT_GOAL := help
.PHONY: help setup eda experiment neural horizons anomaly report all slides test lint
# The pipeline stages read each other's outputs, so they must never run in parallel.
.NOTPARALLEL:

help:
	@printf '\n$(BOLD)Transformer oil-temperature forecasting PoC$(RESET)\n\n'
	@printf '  make test      run the 25 unit tests                       ~2 min the first time, then ~10 s\n'
	@printf '  make all       regenerate every table and figure           ~45 min, shows progress\n'
	@printf '  make slides    rebuild both report PDFs (needs xelatex)    ~1 min\n'
	@printf '  make lint      ruff lint and format check\n\n'
	@printf '  Single stages: make eda | experiment | neural | horizons | anomaly | report\n'
	@printf '  The first command you run sets up .venv automatically (Python 3.12+).\n\n'

# Every target sets up the environment first if needed; `make setup` alone is optional.
setup: $(INSTALLED)

$(INSTALLED): requirements.txt requirements-dev.txt pyproject.toml
	@$(PYTHON) -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null || \
	  { printf 'Python 3.12 or newer is needed (found: %s).\nTry: make PYTHON=python3.12 setup\n' \
	    "$$($(PYTHON) --version 2>&1)"; exit 1; }
	$(call banner,setup,Setting up the Python environment,first run only - about 1-2 min)
	@$(PYTHON) -m venv .venv
	@printf '  … installing pinned packages (numpy, pandas, scikit-learn, torch, ...)\n'
	@$(PIP) -r requirements-dev.txt
	@$(PIP) --no-deps -e .
	@touch $@
	@printf '  $(GREEN)✓$(RESET) environment ready in .venv\n'

eda: $(INSTALLED)
	$(call banner,1/6,Data checks and EDA,about 1 min)
	@$(PY) -m ettpoc.eda
experiment: $(INSTALLED)
	$(call banner,2/6,Main experiment: all models x 12 monthly refits x 2 stations x 2 framings,about 20 min)
	@$(PY) -m ettpoc.experiment
neural: $(INSTALLED)
	$(call banner,3/6,LSTM on the same monthly refits,about 5 min)
	@$(PY) -m ettpoc.neural
horizons: $(INSTALLED)
	$(call banner,4/6,Forecast horizon sweep (1 to 48 h),about 15 min)
	@$(PY) -m ettpoc.horizons
anomaly: $(INSTALLED)
	$(call banner,5/6,Fault-detection study,about 2 min)
	@$(PY) -m ettpoc.anomaly
report: $(INSTALLED)
	$(call banner,6/6,Report tables and figures,about 2 min)
	@$(PY) -m ettpoc.report

all: eda experiment neural horizons anomaly report
	@printf '\n$(GREEN)$(BOLD)✓ All results regenerated.$(RESET)\n'
	@printf '  Tables:   outputs/tables/   (start with results_overall.csv)\n'
	@printf '  Figures:  outputs/figures/\n'
	@printf '  Report:   slides/ETT_PoC_Report_JA.pdf, slides/ETT_PoC_Report_EN.pdf\n'
	@printf '  Next:     make slides   to rebuild the PDFs from the new figures\n\n'

# Report PDFs, Japanese and English (need xelatex and the Hiragino Sans font, i.e. macOS).
XELATEX = xelatex -interaction=nonstopmode -halt-on-error -output-directory=build

slides:
	$(call banner,slides,Building the Japanese and English report PDFs,about 1 min)
	@cd slides && mkdir -p build && for deck in ETT_PoC_Report_JA ETT_PoC_Report_EN; do \
	  { $(XELATEX) $$deck.tex && $(XELATEX) $$deck.tex; } > build/$$deck.console.log 2>&1 \
	    || { printf '  ✗ %s failed; see slides/build/%s.log\n' $$deck $$deck; exit 1; }; \
	  cp build/$$deck.pdf . && printf '  $(GREEN)✓$(RESET) slides/%s.pdf\n' $$deck; done

test: $(INSTALLED)
	$(call banner,test,Unit tests,about 10 s)
	@$(PY) -m pytest -q

lint: $(INSTALLED)
	$(call banner,lint,ruff lint and format check,a few seconds)
	@.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
