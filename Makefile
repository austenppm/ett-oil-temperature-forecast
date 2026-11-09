PY = .venv/bin/python

.PHONY: setup eda experiment neural horizons anomaly report all slides test lint

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pip install --no-deps -e .

eda:        ; $(PY) -m ettpoc.eda
experiment: ; $(PY) -m ettpoc.experiment
neural:     ; $(PY) -m ettpoc.neural
horizons:   ; $(PY) -m ettpoc.horizons
anomaly:    ; $(PY) -m ettpoc.anomaly
report:     ; $(PY) -m ettpoc.report

all: eda experiment neural horizons anomaly report

# Report PDFs, Japanese and English (need xelatex and the Hiragino Sans font, i.e. macOS);
# run after `make all`.
XELATEX = xelatex -interaction=nonstopmode -halt-on-error -output-directory=build

slides:
	cd slides && mkdir -p build && for deck in ETT_PoC_Report_JA ETT_PoC_Report_EN; do \
	  $(XELATEX) $$deck.tex && $(XELATEX) $$deck.tex && cp build/$$deck.pdf . || exit 1; done

test: ; $(PY) -m pytest -q
lint: ; .venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
