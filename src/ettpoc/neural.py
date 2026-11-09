"""Sequence model baseline: a small LSTM on raw windows, same folds and rows as the
main experiment, so its scores are directly comparable.

Inputs at issue time t:
  past 72 h  : OT - OT(t), S^2, six load channels, hour sin/cos   (10 channels)
  next 24 h  : planned S^2 and six load channels                   (framing C only)
  static     : calendar of the target time
Output: dOT = OT(t+24) - OT(t).

Run:  python -m ettpoc.neural        (writes outputs/predictions_nn.csv.gz)
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import config as C
from .data import bad_targets_known_at, clean_ot_causal, load_station, quality_flags, valid_at
from .evaluation import fold_masks, label_cutoff, monthly_folds
from .experiment import Fold
from .features import apparent_power_sq, build, calendar, cyclic
from .progress import done, step, wrote

LOOKBACK = 72
HIDDEN = 32
EPOCHS = 15
PATIENCE = 3
BATCH = 512
LR = 2e-3
TORCH_THREADS = 8  # fixed: the thread count changes float summation order, hence results


class SeqNet(nn.Module):
    def __init__(self, n_past: int, n_future: int, n_static: int):
        super().__init__()
        self.past = nn.LSTM(n_past, HIDDEN, batch_first=True)
        self.future = nn.LSTM(n_future, HIDDEN // 2, batch_first=True) if n_future else None
        width = HIDDEN + (HIDDEN // 2 if n_future else 0) + n_static
        self.head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, past, future, static):
        _, (h, _) = self.past(past)
        parts = [h[-1]]
        if self.future is not None:
            _, (hf, _) = self.future(future)
            parts.append(hf[-1])
        parts.append(static)
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


class Scaler:
    """Standardisation statistics, estimated on the training rows only."""

    def __init__(self, loads: np.ndarray, s2: np.ndarray, d_ot: np.ndarray):
        self.load_mu, self.load_sd = loads.mean(0), loads.std(0) + 1e-6
        self.s2_mu, self.s2_sd = s2.mean(), s2.std()
        self.d_ot_sd = float(np.std(d_ot))  # scales both past OT changes and the target


def model_inputs(df: pd.DataFrame, ot: pd.Series, pos: np.ndarray, framing: str, scaler: Scaler):
    """(past, future, static) arrays for the issue times at hourly positions `pos`.

    `future` is None in framing B.
    """
    loads = (df[C.LOAD_COLS].to_numpy() - scaler.load_mu) / scaler.load_sd
    s2 = ((apparent_power_sq(df).to_numpy() - scaler.s2_mu) / scaler.s2_sd)[:, None]
    hour = np.stack(list(cyclic(df.index.hour.to_numpy(), 24, "hour").values()), axis=1)
    otv = ot.to_numpy()
    base = np.concatenate([otv[:, None], s2, loads, hour], axis=1).astype(np.float32)
    past = base[pos[:, None] + np.arange(-LOOKBACK + 1, 1)[None, :]]
    past[:, :, 0] = (past[:, :, 0] - otv[pos][:, None]) / scaler.d_ot_sd
    future = None
    if framing == "C":
        planned = np.concatenate([s2, loads], axis=1).astype(np.float32)
        future = planned[pos[:, None] + np.arange(1, C.H + 1)[None, :]]
    static = calendar(df.index + pd.Timedelta(hours=C.H)).to_numpy(dtype=np.float32)[pos]
    return past, future, static


def to_tensors(arrays, rows) -> list:
    """Select `rows` of each array and convert to tensors (None stays None)."""
    return [None if a is None else torch.tensor(a[rows]) for a in arrays]


def fit_predict(df, ot, pos_train, y_train, pos_test, framing) -> np.ndarray:
    """Train a SeqNet with early stopping and return its dOT forecasts for `pos_test`.

    Early stopping uses the chronologically last 10 % of the training rows, separated from
    the fitting rows by H rows so that no fitting target falls inside the validation period;
    the scaler is estimated on the fitting rows only.
    """
    n_val = len(pos_train) // 10
    n_fit = len(pos_train) - n_val - C.H
    val = slice(len(pos_train) - n_val, len(pos_train))
    fit = pos_train[:n_fit]
    scaler = Scaler(
        df[C.LOAD_COLS].to_numpy()[fit], apparent_power_sq(df).to_numpy()[fit], y_train[:n_fit]
    )
    inputs = model_inputs(df, ot, pos_train, framing, scaler)
    y = torch.tensor(y_train / scaler.d_ot_sd, dtype=torch.float32)

    past, future, static = inputs
    net = SeqNet(past.shape[2], 0 if future is None else future.shape[2], static.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    best_loss, best_state, epochs_without_gain = np.inf, None, 0
    for _ in range(EPOCHS):
        net.train()
        order = torch.randperm(n_fit)
        for b in range(0, n_fit, BATCH):
            batch = order[b : b + BATCH].numpy()
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(*to_tensors(inputs, batch)), y[batch])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = nn.functional.l1_loss(net(*to_tensors(inputs, val)), y[val]).item()
        if val_loss < best_loss - 1e-4:
            best_loss, epochs_without_gain = val_loss, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    test_inputs = model_inputs(df, ot, pos_test, framing, scaler)
    with torch.no_grad():
        out = net(*to_tensors(test_inputs, slice(None))).numpy()
    return out * scaler.d_ot_sd


def run_station(name: str) -> pd.DataFrame:
    df = load_station(name)
    flags = quality_flags(df)
    ot = clean_ot_causal(df.OT)
    frames = []
    for fr in C.FRAMINGS:
        X, y, meta = build(df, ot, flags, fr)
        pos = df.index.get_indexer(X.index)
        tt = meta.target_time
        usable = (~meta.input_flagged).to_numpy() & (pos >= LOOKBACK)
        step(f"{name} framing {fr}: training an LSTM for each of the 12 months")
        for i, (start, end) in enumerate(monthly_folds(), 1):
            available, test = fold_masks(tt, start, end)
            label_ok = valid_at(bad_targets_known_at(df, label_cutoff(start)), tt)
            train = usable & available.to_numpy() & label_ok
            test = test.to_numpy()
            delta = fit_predict(df, ot, pos[train], y.to_numpy()[train], pos[test], fr)
            frames.append(Fold(name, fr, start, meta[test]).predictions("LSTM", delta))
            done(f"{name} framing {fr}: month {i:2d}/12 ({start:%Y-%m})")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    torch.manual_seed(C.SEED)
    torch.set_num_threads(TORCH_THREADS)
    run_id = C.RUN_ID_FILE.read_text()  # the tabular run these predictions belong to
    out = pd.concat([run_station(n) for n in C.STATIONS], ignore_index=True)
    out.to_csv(C.OUT_DIR / "predictions_nn.csv.gz", index=False)
    C.NN_RUN_ID_FILE.write_text(run_id)
    done(f"Saved {len(out):,} LSTM forecasts")
    wrote(C.OUT_DIR / "predictions_nn.csv.gz")


if __name__ == "__main__":
    main()
