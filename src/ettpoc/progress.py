"""Console progress for the pipeline stages: short, flushed, coloured only on a terminal."""

import sys
import time
from pathlib import Path

from . import config as C

_COLOUR = sys.stdout.isatty()
_START = time.time()


def _paint(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text


def elapsed() -> str:
    seconds = int(time.time() - _START)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def step(message: str) -> None:
    """Something is starting (may take a while)."""
    print(f"  {_paint('2', '…')} {message}", flush=True)


def done(message: str) -> None:
    """Something finished; shows the time since the stage started."""
    print(f"  {_paint('32', '✓')} {message}  {_paint('2', elapsed())}", flush=True)


def note(message: str) -> None:
    print(f"    {_paint('2', message)}", flush=True)


def wrote(*paths: Path) -> None:
    """Where the stage's outputs are, relative to the project root."""
    for path in paths:
        note(f"→ {Path(path).relative_to(C.ROOT)}")
