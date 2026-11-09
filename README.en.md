# Transformer Oil-Temperature Forecasting PoC (ETT dataset)

[日本語](README.md) | **English**

Can a model forecast a power transformer's **oil temperature (OT) 24 hours ahead** well
enough to support preventive maintenance, compared with today's threshold monitoring?

- **Report (client-facing slides):** [English](slides/ETT_PoC_Report_EN.pdf) | [日本語](slides/ETT_PoC_Report_JA.pdf)
- **Code:** `src/ettpoc/` (Python, reproducible with one command)
- **Contact:** austenppm12345.career@gmail.com

**Quick start** (macOS/Linux, Python 3.12+):

```bash
make        # shows the available commands
make test   # first run sets up .venv automatically, then runs the 27 tests (~2 min)
make all    # regenerates every table and figure in 6 numbered stages (~45 min, shows progress)
```

There is no separate setup step: whichever command you run first creates the environment.

## Key results (evaluation year Jul 2017 – Jun 2018, 12 monthly rolling folds)

| | ETTh1 | ETTh2 |
|---|---|---|
| Persistence MAE, 24 h ahead | 1.76 °C | 3.25 °C |
| **Ridge** (physics features + load plan) | **1.60 °C (−9.1 %, Holm-adjusted DM p < 0.001)** | **3.10 °C (−4.4 %, p = 0.003; Holm-adjusted 0.06)** |
| Equal-weight ensemble (C) | 1.65 °C (−6.4 %) | **3.06 °C (−5.8 %, best on ETTh2)** |
| Best skill at shorter lead times (vs. best naive benchmark) | 22 % at 12 h | 53 % at 1–3 h |
| 90 % interval coverage: quantile regression → conformalized (CQR) | 84 % → 90 % | 75 % → 91 % |

- **Early warning (ETTh2, 15 overheating events):** an **Ensemble (C)** alarm that fires when the forecast is within 5 °C of the proxy alarm level warns **12 of 15 events (80 %) at least 12 h ahead** (most about a day ahead), with about **3 false alarms per month** (consecutive alert hours count as one alarm). Today's threshold alarm gives no advance warning. Warnings come either about a day or only 1–2 h before an event, so the share warned ≥12 h ahead is reported rather than a median lead time. The margin was read off the evaluation-year curve, so this is an exploratory operating point, not a validated policy.
- **Complex models did not win.** GBM, LSTM, MLP, a physics + ML hybrid and bias correction did not beat Ridge on ETTh1; on ETTh2 the ensemble leads Ridge by one point. Only Ridge with the plan on ETTh1 and Ridge without the plan on ETTh2 stay significant after a Holm correction across all 25 comparisons with persistence per station (24 model variants plus the 24 h moving average). Our hypothesis is that input information limits accuracy more than model capacity: **ambient temperature is not in the data** and appears to drive the daily cycle.
- **Data quality matters.** Every 31st day of a month is frozen on all channels, ETTh2 has a load sensor stuck for 43 days, and there are ±20 °C single-hour spikes. These are flagged, excluded from scoring, and a stuck load sensor switches the forecast to persistence at run time (a continuity fallback, not necessarily conservative).
- **Fault detection** from the expected-versus-measured OT residual is **not viable without ambient temperature**: synthetic 3–8 °C cooling faults are detected 0 % of the time (0 of 180).

## Problem framing

| | |
|---|---|
| Target | OT at **T = t + 24 h**, forecast issued at hour *t* |
| Framing **B** (strict) | Only data measured up to *t* (plus the calendar of *T*) |
| Framing **C** (what-if) | Also the load profile over (*t*, *T*], interpreted as the operator's **day-ahead load plan**. The brief explicitly allows features at the target time; we state the availability assumption and test its sensitivity to plan error. |
| Model target | Δ OT = OT(*T*) − OT(*t*), which is far more stationary than the level |
| Stations | ETTh1 and ETTh2 (hourly). ETTm1/ETTm2 are the same two transformers at 15-min resolution (their on-the-hour rows equal ETTh1/ETTh2 exactly); hourly resolution is assumed sufficient for a 24 h-ahead decision, so ETTm* were not modelled. |

## Evaluation protocol

- **Development data** = everything observed at the first evaluation issue time (2017-06-30 00:00): hyper-parameter tuning (validation targets from 2017-03-01), alarm thresholds, detector calibration.
- **Evaluation**: 12 monthly **rolling-origin folds** (targets Jul 2017 to Jun 2018), expanding training window. Each month's models are refitted at its **first issue time** (first target − 24 h) using only labels observed by then, then used for the whole month. All seasons are covered, unlike a single spring test split (Tashman 2000; Hewamalage et al. 2023).
- **Benchmarks**: persistence (OT(*T*) = OT(*t*); identical to 24 h seasonal naive at this horizon) and a 24 h moving average. Skill is reported **relative to persistence**, and differences are tested with the **Diebold–Mariano** test (HAC variance, Harvey–Leybourne–Newbold correction) because overlapping 24 h forecasts have autocorrelated errors. p-values are also **Holm-adjusted** across all comparisons per station.
- **No leakage**: preprocessing is fitted inside each training window; whether a training label is an artefact is judged with the data available at the refit; run-time data-quality rules and the online bias correction use only past data. Tests check the fold and tuning cutoffs (`tests/`).
- **Data quality**: targets that are known artefacts (stuck sensors, spikes, the 31st-day fill) are never trained on or scored; forecasts whose inputs come from a stuck load sensor fall back to persistence (as a deployed system would) and carry no prediction interval. Alerts whose ground truth is partly missing are reported as unscorable, not as false.
- **Load columns**: HUFL/HULL etc. are read as high/middle/low "useful"/"useless" load per the dataset's field descriptions; the voltage-side topology (e.g. HUFL = MUFL + LUFL) is an assumption, used only descriptively.

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
| MLP | 2-layer neural network on the engineered features, early-stopped on the chronologically last 10 % of training data |
| WindowLinear | DLinear-style linear model on the raw 168 h window (Zeng et al. 2023) |
| LSTM | Sequence model on the raw past 72 h (+ planned next 24 h for C) |
| Ensemble | Equal-weight mean of Ridge, GBM-robust, Hybrid |
| +BC | Causal bias correction: subtract the model's own mean error over the past 7 days |
| CQR | Conformalized quantile regression (Romano et al. 2019) for 90 % intervals |

## Revision (2026-11-10)

An independent review found a timing leak and several evaluation issues; all were fixed and every
result regenerated:

- Monthly models were trained on targets up to the end of the previous month, i.e. up to 23 h after
  the month's first forecast was issued. They are now refitted at that issue time; tuning, CQR
  calibration, the LSTM and the horizon sweep follow the same rule. Training-row quality labels are
  computed with the data available at the refit.
- Development thresholds no longer include the first evaluation day.
- The load-plan noise study now perturbs only the planned hours (not measured history) and averages
  5 random draws.
- Alerts with incomplete ground truth are unscorable rather than false; synthetic faults use one
  shared, alarm-free onset schedule; the MLP uses chronological early stopping (its earlier
  −45 % result was an artefact of a random hold-out); CQR uses the exact order statistic;
  significance is reported with a Holm correction; the LSTM file is checked to belong to the same run.

A follow-up check found the Ridge and WindowLinear tuning grids too narrow (the chosen values sat at the edge); both were widened so every 24 h choice is interior, and the models rerun. ETTh1's headline is unchanged (−9.1 %); ETTh2 Ridge moved from −4.8 % to −4.4 %. Alerts are now scored as warnings ≥12 h ahead and false alarms per month.

## Reproduce

```bash
make                # list of commands (Python 3.12+, tested on 3.14; the first command creates .venv)
make all            # EDA, experiments, LSTM, horizon sweep, anomaly study, report (~45 min on a laptop CPU)
make test           # unit tests, incl. no-lookahead checks on features, cleaning and flags (~10 s)
make lint           # ruff lint + format check
make slides         # both report PDFs (xelatex + Hiragino Sans, i.e. macOS)
```

Data are downloaded on first run from the upstream ETDataset repository and verified by SHA-256
(they are byte-identical to the files supplied with the assignment). Outputs land in
`outputs/figures/` and `outputs/tables/`; Japanese copies of every figure (used by the Japanese report)
are in `outputs/figures/ja/`.

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
slides/           LaTeX sources of the report, Japanese and English (xelatex), and the PDFs
docs/REFERENCES.md
```

## Data and licence

ETT-small by Zhou et al. (AAAI 2021), CC BY-ND 4.0, https://github.com/zhouhaoyi/ETDataset.
The data are not redistributed in this repository; the code downloads them from the source.
Full reference list: [`docs/REFERENCES.md`](docs/REFERENCES.md).
