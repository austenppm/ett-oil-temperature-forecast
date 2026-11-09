"""Shared matplotlib style so every figure in the report looks consistent."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C

STATION_COLORS = {"ETTh1": "#2b6cb0", "ETTh2": "#c05621"}
MODEL_COLORS = {
    "Persistence": "#8a8a8a",
    "MovingAvg24": "#b5b5b5",
    "GreyBox": "#2f855a",
    "Ridge": "#2b6cb0",
    "GBM": "#c05621",
}
INK = "#222222"
MUTED = "#6b6b6b"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#999999",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / f"{name}.png")
    plt.close(fig)
