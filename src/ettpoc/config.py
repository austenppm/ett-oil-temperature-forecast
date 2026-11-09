"""Project-wide constants: paths, data sources, experiment design."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
# Run identity: written by the experiment, copied by the LSTM run, checked by the report.
RUN_ID_FILE = OUT_DIR / "run_id.txt"
NN_RUN_ID_FILE = OUT_DIR / "predictions_nn.run_id.txt"

# Upstream source of the ETT-small data (CC BY-ND 4.0, Zhou et al., AAAI 2021).
# The files supplied with the assignment are byte-identical to these.
UPSTREAM_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
SHA256 = {
    "ETTh1": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
    "ETTh2": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
}
STATIONS = ["ETTh1", "ETTh2"]
LOAD_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

# Forecast design: issue time t, target time T = t + H hours.
H = 24
# B: strict (data up to t only); C: what-if, with the day-ahead load plan over (t, T].
FRAMINGS = ("B", "C")
# Load-plan error levels (relative sigma) for the framing-C robustness study.
NOISE_SIGMAS = (0.05, 0.10, 0.20)

# Chronology. Test windows are defined by TARGET time; a model is fitted at the first
# issue time of its window (target start - H) using only labels observed by then.
TUNE_VAL_START = "2017-03-01"  # tuning: validate on targets [this, EVAL_START)
EVAL_START = "2017-07-01"  # rolling-origin evaluation: 12 monthly folds
EVAL_END = "2018-06-27"
# Development data = everything observed at the first evaluation issue time. Thresholds,
# alert levels and calibration use only these hours.
DEV_END = pd.Timestamp(EVAL_START) - pd.Timedelta(hours=H)

# Data-quality rules
STUCK_LOAD_HOURS = 24  # a load channel unchanged for >= 24 h is treated as a stuck sensor
STUCK_OT_HOURS = 12  # oil temperature unchanged for >= 12 h is implausible
SPIKE_DEG = 6.0  # |OT - local median| above this (degC) is a single-point glitch

# Alerting: proxy alarm level = this quantile of development-period OT (no real limit supplied)
ALERT_QUANTILE = 0.95
INTERVAL_ALPHA = 0.10  # 90 % prediction intervals

SEED = 42
