import pandas as pd

from ettpoc import config as C
from ettpoc import report


def test_example_forecast_figure_uses_current_alert_inputs(monkeypatch, tmp_path, station):
    monkeypatch.setattr(report, "load_station", lambda _name: station)
    monkeypatch.setattr(C, "FIG_DIR", tmp_path)
    times = pd.date_range("2017-07-08", periods=4, freq="h")
    records = [
        {
            "station": "ETTh2",
            "framing": framing,
            "model": model,
            "target_time": target,
            "ot_true": 20.0 + i,
            "ot_pred": 19.5 + i,
            "lo": 18.0 + i,
            "hi": 23.0 + i,
        }
        for framing, model in (("C", "GBM"), ("C", "Ridge"), ("-", "Persistence"))
        for i, target in enumerate(times)
    ]

    report.fig_example(pd.DataFrame(records))

    assert (tmp_path / "res_example_forecast.png").is_file()
