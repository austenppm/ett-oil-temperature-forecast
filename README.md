# Transformer Oil-Temperature Forecasting PoC (ETT dataset)

Can a model forecast a power transformer's **oil temperature (OT) 24 hours ahead** well
enough to support preventive maintenance, compared with today's threshold monitoring?

- **Report (client-facing slides, Japanese):** [`slides/ETT_PoC_Report.pdf`](slides/ETT_PoC_Report.pdf)
- **Code:** `src/ettpoc/` (Python, reproducible with one command)
- **Contact:** austenppm12345.career@gmail.com

## Key results (evaluation year Jul 2017 – Jun 2018, 12 monthly rolling folds)

| | ETTh1 | ETTh2 |
|---|---|---|
| Persistence MAE, 24 h ahead | 1.76 °C | 3.25 °C |
| **Best model** (Ridge, physics features + load plan) | **1.60 °C (−9.1 %, DM p < 0.001)** | **3.09 °C (−4.8 %, p = 0.005)** |
| Best skill at shorter lead times (vs. best naive benchmark) | 22 % at 12 h | 53 % at 1–3 h |
| 90 % interval coverage: quantile regression → conformalized (CQR) | 84 % → 90 % | 77 % → 90 % |

- **Early warning (ETTh2, 15 hot-spell onsets):** alarming when the forecast is within 4 °C of the proxy alarm level warns **87 % of onsets a median 21 h ahead** with ~11 false alert-hours per month. Today's threshold alarm gives 0 h warning.
- **Complex models did not win.** GBM, LSTM, MLP, a physics + ML hybrid and bias correction all failed to beat the linear model at 24 h. The input information is the bottleneck, not model capacity: **ambient temperature is not in the data** and appears to drive the daily cycle.
- **Data quality matters.** Every 31st day of a month is frozen on all channels, ETTh2 has a load sensor stuck for 43 days, and there are ±20 °C single-hour spikes. These are flagged, excluded from scoring, and handled at run time by falling back to persistence.
- **Fault detection** from the expected-versus-measured OT residual is **not viable without ambient temperature**: synthetic 3–8 °C cooling faults are detected ≤ 10 % of the time.

## Problem framing

| | |
|---|---|
| Target | OT at **T = t + 24 h**, forecast issued at hour *t* |
| Framing **B** (strict) | Only data measured up to *t* (plus the calendar of *T*) |
| Framing **C** (what-if) | Also the load profile over (*t*, *T*], interpreted as the operator's **day-ahead load plan**. The brief explicitly allows features at the target time; we state the availability assumption and test its sensitivity to plan error. |
| Model target | Δ OT = OT(*T*) − OT(*t*), which is far more stationary than the level |
| Stations | ETTh1 and ETTh2 (hourly). ETTm* are the same series at 15-min resolution and gave identical conclusions. |

## Evaluation protocol

- **Development data** (targets before 2017-07-01): hyper-parameter tuning (train < 2017-03, validate 2017-03 to 06), alarm thresholds, detector calibration.
- **Evaluation**: 12 monthly **rolling-origin folds** (Jul 2017 to Jun 2018), expanding training window, every model refit at each origin. All seasons are covered, unlike a single spring test split (Tashman 2000; Hewamalage et al. 2023).
- **Benchmarks**: persistence (OT(*T*) = OT(*t*); identical to 24 h seasonal naive at this horizon) and a 24 h moving average. Skill is reported **relative to persistence**, and differences are tested with the **Diebold–Mariano** test (HAC variance, Harvey–Leybourne–Newbold correction) because overlapping 24 h forecasts have autocorrelated errors.
- **No leakage**: all preprocessing is fitted inside each training window; run-time data-quality rules use only past data.
- **Data quality**: targets that are known artefacts (stuck sensors, spikes, the 31st-day fill) are never trained on or scored; forecasts whose inputs come from a stuck load sensor fall back to persistence (as a deployed system would).

## Models

| Model | Idea |
|---|---|
| Persistence, MovingAvg24 | Benchmarks |
| GreyBox | Discretised IEC 60076-7 first-order top-oil model; time constant τ and load gain learned from data; ambient (unmeasured) approximated by calendar harmonics and weekly mean OT |
| Ridge | Linear model on physics-motivated features (S² = useful² + useless² ∝ current², EWMAs of S² at 3/12/48 h half-lives, OT dynamics, calendar) |
| GBM | Histogram gradient boosting (LightGBM-style) on the same features |
| GBM-robust | GBM without the absolute OT level, with recency weighting (half-life 180 days), against year-to-year drift |
| Hybrid | GreyBox + GBM-robust trained on the physics model's residual |
| GBM-peak | GBM-robust with hot targets (top 10 %) up-weighted ×3 |
| MLP | 2-layer neural network on the engineered features |
| WindowLinear | DLinear-style linear model on the raw 168 h window (Zeng et al. 2023) |
| LSTM | Sequence model on the raw past 72 h (+ planned next 24 h for C) |
| Ensemble | Equal-weight mean of Ridge, GBM-robust, Hybrid |
| +BC | Causal bias correction: subtract the model's own mean error over the past 7 days |
| CQR | Conformalized quantile regression (Romano et al. 2019) for 90 % intervals |

## Reproduce

```bash
make setup          # Python 3.11+; creates .venv, installs pinned requirements and the package
make all            # EDA, experiments, LSTM, horizon sweep, anomaly study, report (~45 min on a laptop CPU)
make test           # unit tests, incl. no-lookahead checks on features, cleaning and flags (~10 s)
make lint           # ruff lint + format check
```

Data are downloaded on first run from the upstream ETDataset repository and verified by SHA-256
(they are byte-identical to the files supplied with the assignment). Outputs land in
`outputs/figures/` and `outputs/tables/`.

```
src/ettpoc/
  config.py       paths, data source, chronology, rules
  data.py         loading, SHA-256 check, data-quality flags (retrospective + causal)
  features.py     feature construction for framings B and C, raw windows
  models.py       Ridge / GBM / MLP factories, grey-box thermal model, CQR
  evaluation.py   metrics, Diebold–Mariano, alert simulation
  eda.py          EDA figures and summary table
  experiment.py   tuning + 12-fold rolling-origin evaluation of all tabular models
  neural.py       LSTM on raw windows (same folds and rows)
  horizons.py     1–48 h horizon sweep
  anomaly.py      residual-based fault detection study
  report.py       ensemble, bias correction, tables and figures for the report
tests/            pytest: leakage (features ignore the future), data-quality rules,
                  Diebold–Mariano, alert episodes, CQR coverage, grey-box tau recovery
slides/           LaTeX source of the report (xelatex) and the compiled PDF
docs/REFERENCES.md
```

## Data and licence

ETT-small by Zhou et al. (AAAI 2021), CC BY-ND 4.0, https://github.com/zhouhaoyi/ETDataset.
The data are not redistributed in this repository; the code downloads them from the source.
Full reference list: [`docs/REFERENCES.md`](docs/REFERENCES.md).
