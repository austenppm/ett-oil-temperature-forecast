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

# Report PDF (needs xelatex and the Hiragino Sans font, i.e. macOS); run after `make all`.
slides:
	cd slides && mkdir -p build && xelatex -interaction=nonstopmode -halt-on-error -output-directory=build ETT_PoC_Report.tex \
	  && xelatex -interaction=nonstopmode -halt-on-error -output-directory=build ETT_PoC_Report.tex && cp build/ETT_PoC_Report.pdf .

test: ; $(PY) -m pytest -q
lint: ; .venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
