"""Shared matplotlib style so every figure in the report looks consistent."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

from . import config as C
from .figure_text_ja import translate

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


JA_FONT = ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "DejaVu Sans"]


def save(fig, name):
    """Save the English figure, then a Japanese copy (outputs/figures/ja/) for the JA report."""
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / f"{name}.png")
    missing = set()
    fig.canvas.draw()  # materialise tick labels
    # Category tick labels are regenerated on save, so replace them explicitly.
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labels = [t.get_text() for t in axis.get_ticklabels()]
            translated = [translate(label)[0] for label in labels]
            if translated != labels:
                axis.set_ticks(axis.get_ticklocs(), translated)
    for text in fig.findobj(Text):
        if not text.get_text().strip():
            continue
        already_japanese = any(ch >= "\u3000" for ch in text.get_text())
        ja, known = (text.get_text(), True) if already_japanese else translate(text.get_text())
        if not known:
            missing.add(text.get_text())
        text.set_text(ja)
        text.set_fontfamily(JA_FONT)
    if missing:
        print(f"  ! {name}: no Japanese text for {sorted(missing)}", flush=True)
    ja_dir = C.FIG_DIR / "ja"  # resolved per call, so a redirected FIG_DIR is respected
    ja_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(ja_dir / f"{name}.png")
    plt.close(fig)
