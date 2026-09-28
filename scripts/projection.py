"""Projected compute cost of the confirmatory plan, per model.

No forecast of Labour or M5 is made. Their files are read only to count series
and dates. The models are timed on synthetic series of the same length and
horizon, and the timing is calibrated on the measured TourismLarge origin
(docs/feasibility.json).

    cost = calls x series x seconds per series(context length, horizon)

Plan: TourismLarge 5 origins, Labour 5 origins (the 2 confirmatory ones are
among the 5 registered ones), M5 5 origins with h = 28. Each origin is one base
forecast and 24 backtest calls; chronos_t5_small is base only.

Writes docs/projection.md and docs/projection.json.

Run from the repo root: python scripts/projection.py [--report]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import DATASETS, load_hierarchy, load_m5  # noqa: E402
from src.forecasters import forecast_cutoffs  # noqa: E402
from src.residuals import inner_origins  # noqa: E402
from src.runners import MODELS  # noqa: E402

K, N_INNER = 5, 24
LIMIT_HOURS = 8.0
N_TIMED = 64
BASE_ONLY = {"chronos_t5_small"}
CONTEXT_LIMIT = {"chronos_bolt_tiny": 2048, "chronos_bolt_mini": 2048, "chronos_bolt_small": 2048,
                 "chronos_t5_small": 512, "chronos_2": 8192, "tirex": 2016, "moirai_2_small": None}
MODELS_ORDER = ["chronos_bolt_tiny", "chronos_bolt_mini", "chronos_bolt_small", "chronos_t5_small",
                "chronos_2", "tirex", "moirai_2_small"]
OUT_JSON, OUT_MD = ROOT / "docs" / "projection.json", ROOT / "docs" / "projection.md"
FEASIBILITY = ROOT / "docs" / "feasibility.json"


def describe(name, Y_df, S_df, tags, freq, h, note=""):
    """Series, dates and the contexts of every forecast call. Reads no values."""
    dates = pd.DatetimeIndex(sorted(Y_df["ds"].unique()))
    T = len(dates)
    origins, base_ctx, bt_ctx = [], [], []
    for k in range(K, 0, -1):
        n_train = T - h * k
        origins.append({"origin": str(dates[n_train - 1].date()), "train_len": int(n_train),
                        "test": f"{dates[n_train].date()} to {dates[n_train + h - 1].date()}"})
        base_ctx.append(n_train)
        inner = inner_origins(dates[:n_train], h, N_INNER)
        bt_ctx += [int((dates <= s).sum()) for s in inner]
    return {
        "dataset": name, "note": note, "freq": freq, "h": h,
        "series": int(S_df.shape[0]), "bottom": int(S_df.shape[1] - 1),
        "levels": {k2: int(len(v)) for k2, v in tags.items()},
        "first": str(dates[0].date()), "last": str(dates[-1].date()), "periods": T,
        "origins": origins,
        "context_base": [int(min(base_ctx)), int(max(base_ctx))],
        "context_backtest": [int(min(bt_ctx)), int(max(bt_ctx))],
        "mean_context_all_calls": float(np.mean(base_ctx + bt_ctx)),
        "mean_context_base": float(np.mean(base_ctx)),
    }


def synthetic(n, length, freq, seed=0):
    """Positive seasonal series with noise. Only their shape matters for timing."""
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2000-01-01", periods=length, freq=freq)
    t = np.arange(length)
    m = 12 if freq == "MS" else 7
    rows = []
    for i in range(n):
        y = 100 + 20 * np.sin(2 * np.pi * t / m) + rng.normal(0, 5, length) + 0.01 * t
        rows.append(pd.DataFrame({"unique_id": f"s{i}", "ds": ds, "y": np.maximum(y, 0.0)}))
    return pd.concat(rows, ignore_index=True)


def time_shape(model, length, h, freq):
    """Seconds per series for one call on synthetic series of this shape."""
    y = synthetic(N_TIMED, length, freq)
    cutoff = y["ds"].max()
    warm = y[y["unique_id"].isin(["s0", "s1"])]
    # Both calls in one go for models in another env, so the model is loaded once
    _, info = forecast_cutoffs(model, warm, [cutoff], h, freq=freq)
    times = []
    for _ in range(2):
        _, info = forecast_cutoffs(model, y, [cutoff], h, freq=freq)
        times.append(info["seconds"][0])
    return float(min(times)) / N_TIMED


def main():
    report_only = "--report" in sys.argv
    if report_only:
        write_markdown(json.loads(OUT_JSON.read_text()))
        return

    tl = describe("TourismLarge", *load_hierarchy("TourismLarge"))
    lab = describe("Labour", *load_hierarchy("Labour"),
                   note="confirmatory: the test years 2018 and 2019; exploratory: 2015 to 2019")
    m5a = describe("M5, 1 store (CA_1)", *load_hierarchy("M5"))
    Y2, S2, tags2 = load_m5(("CA_1", "CA_2"))
    m5b = describe("M5, 2 stores (CA_1, CA_2)", Y2, S2, tags2, DATASETS["M5"]["freq"], DATASETS["M5"]["h"])
    data = [tl, lab, m5a, m5b]
    shapes = {d["dataset"]: (int(round(d["mean_context_all_calls"])), int(round(d["mean_context_base"])),
                             d["h"], d["freq"]) for d in data}

    measured = {r["model"]: r["timing"] for r in json.loads(FEASIBILITY.read_text()) if r.get("timing")}
    # chronos_bolt_small was timed in Week 3
    bs_base = pd.read_parquet(ROOT / "results" / "timing_TourismLarge_2015-12-01_chronos_bolt_small.parquet")
    bs_bt = pd.read_parquet(ROOT / "results" / "backtest_timing_TourismLarge_2015-12-01_chronos_bolt_small.parquet")
    measured["chronos_bolt_small"] = {"base_seconds": float(bs_base["seconds"].iloc[0]),
                                      "backtest_seconds": float(bs_bt["seconds"].iloc[0])}

    records = []
    for model in MODELS_ORDER:
        rec = {"model": model, "env": MODELS[model]["env"], "base_only": model in BASE_ONLY,
               "context_limit": CONTEXT_LIMIT[model], "per_series": {}, "cost": {}}
        calls_per_origin = 1 if model in BASE_ONLY else 1 + N_INNER
        t0 = time.perf_counter()
        for d in data:
            ctx_all, ctx_base, h, freq = shapes[d["dataset"]]
            ctx = ctx_base if model in BASE_ONLY else ctx_all
            key = (ctx, h, freq)
            rec["per_series"][d["dataset"]] = {"context": ctx, "h": h}
            cached = [v for v in rec["per_series"].values() if v.get("key") == list(key) and "synthetic" in v]
            sec = cached[0]["synthetic"] if cached else time_shape(model, ctx, h, freq)
            rec["per_series"][d["dataset"]].update({"key": list(key), "synthetic": sec})

        # calibration: measured TourismLarge seconds per series and call, against the synthetic estimate
        m = measured[model]
        if model in BASE_ONLY:
            measured_ps = m["base_seconds"] / tl["series"]
        else:
            measured_ps = (m["base_seconds"] + m["backtest_seconds"]) / (1 + N_INNER) / tl["series"]
        rec["measured_tourism_per_series"] = measured_ps
        rec["calibration"] = measured_ps / rec["per_series"]["TourismLarge"]["synthetic"]
        for d in data:
            ps = rec["per_series"][d["dataset"]]
            ps["projected"] = ps["synthetic"] * rec["calibration"]
            ps["ratio_to_tourism"] = ps["synthetic"] / rec["per_series"]["TourismLarge"]["synthetic"]
            calls = K * calls_per_origin
            rec["cost"][d["dataset"]] = {"calls": calls, "series": d["series"],
                                         "seconds": calls * d["series"] * ps["projected"]}
        c = rec["cost"]
        rec["total_1_store"] = c["TourismLarge"]["seconds"] + c["Labour"]["seconds"] + c[m5a["dataset"]]["seconds"]
        rec["total_2_stores"] = c["TourismLarge"]["seconds"] + c["Labour"]["seconds"] + c[m5b["dataset"]]["seconds"]
        rec["over_limit_1_store"] = bool(rec["total_1_store"] > LIMIT_HOURS * 3600)
        rec["over_limit_2_stores"] = bool(rec["total_2_stores"] > LIMIT_HOURS * 3600)
        rec["timing_seconds"] = time.perf_counter() - t0
        records.append(rec)
        print(f"{model}: total {rec['total_1_store'] / 3600:.2f} h with 1 M5 store, "
              f"{rec['total_2_stores'] / 3600:.2f} h with 2; calibration {rec['calibration']:.2f}", flush=True)
        OUT_JSON.write_text(json.dumps({"data": data, "models": records}, indent=1))

    out = {"data": data, "models": records, "limit_hours": LIMIT_HOURS, "n_timed": N_TIMED}
    OUT_JSON.write_text(json.dumps(out, indent=1))
    write_markdown(out)


def hours(sec):
    return f"{sec / 3600:.2f} h" if sec >= 3600 else f"{sec / 60:.1f} min"


def write_markdown(out):
    data, models = out["data"], out["models"]
    names = [d["dataset"] for d in data]
    drows = ["| Dataset | Series | Bottom series | Levels | Range | Periods | h | History at the base forecasts | "
             "History at the backtest calls |", "|" + "---|" * 9]
    for d in data:
        drows.append(f"| {d['dataset']} | {d['series']:,} | {d['bottom']:,} | "
                     + ", ".join(f"{v:,}" for v in d["levels"].values())
                     + f" | {d['first']} to {d['last']} | {d['periods']:,} | {d['h']} | "
                     f"{d['context_base'][0]:,} to {d['context_base'][1]:,} | "
                     f"{d['context_backtest'][0]:,} to {d['context_backtest'][1]:,} |")
    orows = ["| Dataset | Origin (last training period) | Test window |", "|---|---|---|"]
    for d in data[:3]:
        for o in d["origins"]:
            orows.append(f"| {d['dataset']} | {o['origin']} | {o['test']} |")

    crows = ["| Model | TourismLarge | Labour | M5, 1 store | M5, 2 stores | Total, 1 store | Total, 2 stores | "
             f"Over {out['limit_hours']:g} h |", "|" + "---|" * 8]
    srows = ["| Model | Context limit | " + " | ".join(f"{n}: s per series" for n in names)
             + " | M5 against TourismLarge, per series | Calibration |", "|" + "---|" * (len(names) + 4)]
    for r in models:
        c = r["cost"]
        over = ("yes" if r["over_limit_1_store"] else
                ("only with 2 stores" if r["over_limit_2_stores"] else "no"))
        label = r["model"] + (" (base only)" if r["base_only"] else "")
        crows.append(f"| {label} | " + " | ".join(hours(c[n]["seconds"]) for n in names)
                     + f" | {hours(r['total_1_store'])} | {hours(r['total_2_stores'])} | {over} |")
        ps = r["per_series"]
        srows.append(f"| {r['model']} | {r['context_limit'] or 'none'} | "
                     + " | ".join(f"{ps[n]['projected']:.4f}" for n in names)
                     + f" | {ps[names[2]]['ratio_to_tourism']:.1f} times | {r['calibration']:.2f} |")

    shapes = ["| Dataset | Calls per model | Series per call | Mean history over the calls | h |",
              "|---|---|---|---|---|"]
    for d in data:
        shapes.append(f"| {d['dataset']} | {K} x {1 + N_INNER} = {K * (1 + N_INNER)} "
                      f"(base only: {K}) | {d['series']:,} | {d['mean_context_all_calls']:.0f} | {d['h']} |")

    text = f"""# Projected compute cost of the confirmatory plan

Generated by `scripts/projection.py`. Do not edit by hand.

No forecast of Labour or M5 was made. Their files were read only to count series and dates. The models
were timed on {out['n_timed']} synthetic series of the same length and horizon.

## Datasets

{chr(10).join(drows)}

Labour: the 2 confirmatory origins (test years 2018 and 2019) are 2 of the 5 registered origins, so
Labour needs 5 origins of forecasts, not 7.

M5: PREREG §3 does not name the stores. CA_1 and CA_2 are the first two in the order of the M5 files.
Days d_1 to d_1969. Leading zeros are kept, so every series has all 1,969 days. With 2 stores there is
no level above the stores.

## Origins

{chr(10).join(orows)}

## Projected cost

{chr(10).join(crows)}

cost = calls x series x seconds per series. CPU, one model at a time.

## Scaling

{chr(10).join(shapes)}

{chr(10).join(srows)}

- Seconds per series are for one forecast call, after calibration.
- Calibration is the measured TourismLarge time per series (docs/feasibility.json) divided by the
  synthetic estimate for the same shape. It corrects for batch size and overhead; it is applied to
  every dataset.
- A model with a context limit uses only the last part of a longer history, so its cost stops growing
  with the history beyond that limit.
"""
    OUT_MD.write_text(text)
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
