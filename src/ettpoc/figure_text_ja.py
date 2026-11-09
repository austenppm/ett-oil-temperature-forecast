# ruff: noqa: E501  (a table of phrases reads better unwrapped)
"""Japanese text for the figures of the Japanese report.

Figures are drawn once in English; `plotting.save` then translates every text element with
`translate` and saves a copy under outputs/figures/ja/. Model names (Ridge, GBM, LSTM, CQR)
stay in English, as in the Japanese slides.
"""

import re

MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        1,
    )
}

EXACT = {
    # EDA
    "30-day mean": "30日移動平均",
    "Evaluation year": "評価年",
    "OT": "油温",
    "Hour of day": "時刻",
    "Load S² (z)": "負荷 S²（標準化）",
    "OT deviation": "油温の偏差",
    "Half-life of load smoothing (hours, log scale)": "負荷の平滑化の半減期（時間、対数軸）",
    "Oil temperature responds to load with thermal memory": "油温は時間遅れを伴って負荷に反応する",
    "corr(OT, smoothed S²)": "相関（油温, 平滑化 S²）",
    "OT spike": "油温スパイク",
    "OT stuck": "油温固着",
    "Load balance check: high-voltage load should equal middle + low load": (
        "負荷バランスの確認：高圧側負荷＝中圧＋低圧（列名からの仮定）"
    ),
    "Forecast horizon (hours, log scale)": "予測の先行時間（時間、対数軸）",
    "MAE of 'no change' forecast": "「現在値維持」のMAE",
    "Persistence error grows with the horizon (12 h is out of phase)": (
        "「現在値維持」の誤差は先行時間とともに増加（12時間先は日周期と逆位相）"
    ),
    # headline and skill
    "24 h error reduction vs persistence [%]": "「現在値維持」比の誤差削減率 [%]",
    "Ridge (B), no plan": "Ridge（B）計画なし",
    "Ridge (C), with plan": "Ridge（C）計画あり",
    "Ensemble (C), with plan": "アンサンブル（C）計画あり",
    "How much lower the 24 h-ahead error is than persistence (* Holm-adjusted p < 0.05)": (
        "24時間先の誤差は「現在値維持」よりどれだけ小さいか（* Holm補正後 p < 0.05）"
    ),
    "24 h-ahead forecast skill, 12 monthly rolling folds "
    "(* = Diebold–Mariano p < 0.05 vs persistence, Holm-adjusted)": (
        "24時間先予測のスキル（12か月ロール検証、* = Holm補正後のDiebold–Mariano検定 p < 0.05）"
    ),
    "B: strict (data up to t)": "B：厳密（時点tまでのデータ）",
    "C: with day-ahead load plan": "C：翌日の負荷計画あり",
    "MAE reduction vs persistence [%]  (clipped at −15)": "「現在値維持」比のMAE削減率 [%]（−15で打ち切り）",
    "Ensemble": "アンサンブル",
    "Ensemble+BC": "アンサンブル+BC",
    "Persistence": "現在値維持",
    "MovingAvg24": "24時間移動平均",
    "GreyBox": "グレーボックス",
    "Hybrid": "ハイブリッド",
    # error analysis
    "By month: gains vary; a few months are worse": "月別：改善は月によりばらつき、悪化する月もある",
    "ETTh2 by temperature: worse than persistence when hottest": (
        "ETTh2 温度帯別：最高温域では「現在値維持」より誤差が大きい"
    ),
    "Error reduction vs persistence [%]": "「現在値維持」比の誤差削減率 [%]",
    "Measured oil temperature at the target time [°C]": "予測対象時刻の実測油温 [°C]",
    "Ensemble (C)": "アンサンブル（C）",
    # coverage
    "Coverage of the 90 % interval by month (framing C)": "90%区間の月別被覆率（計画あり C）",
    "Truths inside the interval [%]": "区間内に入った実測の割合 [%]",
    "target 90 %": "目標 90%",
    # alerts
    "False alarms per month (consecutive alert hours count as one alarm)": (
        "月あたりの誤報回数（連続する警告時間は1回と数える）"
    ),
    "Hot spells warned ≥12 h ahead [%]": "12時間以上前に警告できた過熱の割合 [%]",
    "Today's threshold alarm (no advance warning)": "現行の閾値監視（事前警告なし）",
    "★ alarm on the 90 % upper bound (CQR)": "★ 90%区間の上限で警告（CQR）",
    # load plan
    "Error of the day-ahead load plan (σ, % of load)": "翌日の負荷計画の誤差（σ、負荷に対する%）",
    "Skill vs persistence [%]": "「現在値維持」比のスキル [%]",
    "The load plan's value depends on its accuracy": "負荷計画の価値はその精度に依存する",
    # importance
    "Calendar": "カレンダー",
    "OT history": "油温の履歴",
    "Past loads": "過去の負荷",
    "Planned loads (t,T]": "計画負荷 (t,T]",
    "MAE increase when the feature group is shuffled": "特徴量グループを並べ替えたときのMAE増加",
    # example forecast
    "90 % interval (CQR)": "90%区間（CQR）",
    "Measured OT": "実測油温",
    "Ridge (C), issued 24 h earlier": "Ridge（C）、24時間前に発行",
    # horizons
    "Forecast horizon": "予測の先行時間",
    "GBM-robust (C)": "頑健化GBM（C）",
    "Skill vs best naive benchmark [%]": "最良の単純予測比のスキル [%]",
    # anomaly
    "6 h mean gap\n(measured − expected)": "差の6時間平均\n（実測 − 期待値）",
    "Expected OT from loads": "負荷から推定した期待油温",
    "Fault active": "故障発生中",
    "no alarm while the fault is active": "故障中に警報なし",
}

PATTERNS = [
    (r"^(ETTh\d): oil temperature \(OT\)$", r"\1：油温（OT）"),
    (
        r"^(ETTh\d): OT peaks at (\d+):00, load at (\d+):00$",
        r"\1：油温のピーク \2時、負荷のピーク \3時",
    ),
    (
        r"^(ETTh\d): ([\d.]+)% of hours with a stuck load channel; (\d+) days "
        r"\(all on the 31st\) frozen on every channel$",
        r"\1：負荷センサ固着 \2%（時間比）、全項目の凍結 \3日（すべて31日）",
    ),
    (r"^(\w+) stuck$", r"\1 固着"),
    (r"^>(\d+)\n\(hottest 10 %\)$", r">\1\n（最高温10%）"),
    (r"^(ETTh\d): after CQR$", r"\1：CQR補正後"),
    (r"^(ETTh\d): quantile regression$", r"\1：分位点回帰（補正前）"),
    (
        r"^(ETTh\d): early-warning trade-off \(proxy alarm level ([\d.]+)\)$",
        r"\1：事前警告のトレードオフ（代理警報値 \2℃）",
    ),
    (r"^Ensemble \((\w)\) \+ margin 0–10 °C$", r"アンサンブル（\1）＋余裕幅 0〜10℃"),
    (r"^(ETTh\d) without plan \(B\)$", r"\1 計画なし（B）"),
    (
        r"^(ETTh\d): Ridge \(C\), mean of (\d+) plan-error draws$",
        r"\1：Ridge（C）、計画誤差 \2回の平均",
    ),
    (r"^(ETTh\d): what GBM \(C\) relies on$", r"\1：GBM（C）が依存する情報"),
    (r"^(ETTh\d): value of the model by lead time$", r"\1：先行時間別のモデルの価値"),
    (
        r"^(ETTh\d): the expected OT follows an injected \+(\d+) °C cooling fault$",
        r"\1：期待油温が擬似冷却故障（+\2℃）に追従する",
    ),
    (r"^Measured OT \(with injected \+(\d+) °C fault\)$", r"実測油温（+\1℃の擬似故障を注入）"),
    (r"^alarm (\d+) h after onset$", r"故障発生から\1時間後に警報"),
    (r"^(\d+) °C$", r"\1℃"),
]

# Text that is the same in both languages (station and model names, numbers, dates).
UNCHANGED = re.compile(
    r"^(ETTh\d(: r = [\d.]+)?|ETTh\d: Ridge \(C\)|Ridge( \([BC]\))?|Ridge\+BC|GBM(-peak|-robust)?|"
    r"MLP|LSTM|WindowLinear|HUFL|MUFL \+ LUFL|MAE \[°C\]|[<>]?\d+(–\d+)?|"
    r"[−+\-]?[\d.,]+%?\*?|\d+h|20\d\d-\d\d(-\d\d)?)$"
)


def _example_title(text: str) -> str | None:
    m = re.match(
        r"^(ETTh\d), (\w+) (\d{4}) \(median month: ([+−-]?\d+%) vs persistence\), "
        r"24 h-ahead forecasts at their target time$",
        text,
    )
    if not m:
        return None
    station, month, year, skill = m.groups()
    return f"{station}、{year}年{MONTHS[month]}月（改善率が中央値の月：{skill}）の24時間先予測"


def translate(text: str) -> tuple[str, bool]:
    """(Japanese text, whether it was recognised). Leading/trailing spaces are kept."""
    core = text.strip()
    lead, trail = text[: len(text) - len(text.lstrip())], text[len(text.rstrip()) :]
    if core in EXACT:
        return lead + EXACT[core] + trail, True
    if (title := _example_title(core)) is not None:
        return lead + title + trail, True
    for pattern, replacement in PATTERNS:
        if re.match(pattern, core, flags=re.S):
            return lead + re.sub(pattern, replacement, core, flags=re.S) + trail, True
    return text, bool(UNCHANGED.match(core))
