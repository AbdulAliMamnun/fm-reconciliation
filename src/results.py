"""The long results table (PREREG §10).

One row per (dataset, origin, model, W_est, series_id, horizon). Metrics are
computed from this table afterwards (see src/metrics.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILE_COLS = [f"q{q}" for q in range(10, 100, 10)]
KEY_COLS = ["dataset", "origin", "model", "W_est"]
# `rmsse_scale` is not in the PREREG §10 schema. It holds the mean squared
# seasonal-naive error in-sample, which RMSSE needs and `mase_scale` cannot give.
COLUMNS = [
    "dataset", "origin", "model", "W_est", "level", "series_id", "horizon",
    "y", "yhat", *QUANTILE_COLS, "mase_scale", "rmsse_scale",
]


def build_results(dataset, origin, model, forecasts, actuals, tags, scales, w_est,
                  quantiles=None):
    """Put one origin's forecasts into the results-table format.

    forecasts: long frame with unique_id, ds and one column per forecast.
    actuals: long frame with unique_id, ds, y covering the forecast dates.
    tags: dict mapping level name -> array of series ids.
    scales: frame with unique_id, mase_scale, rmsse_scale (training data only).
    w_est: dict mapping forecast column -> W_est label.
    quantiles: optional long frame with unique_id, ds, W_est (the label) and
        q10..q90. Rows of the table without a match keep NaN quantiles.

    `origin` is the last training timestamp. `horizon` counts from 1.
    """
    missing = [c for c in w_est if c not in forecasts.columns]
    if missing:
        raise ValueError(f"Forecast columns not found: {missing}")

    level_of = {uid: level for level, ids in tags.items() for uid in ids}
    order = {uid: i for i, uid in enumerate(level_of)}
    unknown = set(forecasts["unique_id"]) - set(level_of)
    if unknown:
        raise ValueError(f"Series not in tags, e.g. {sorted(unknown)[:3]}")

    dates = np.sort(forecasts["ds"].unique())
    if pd.Timestamp(dates[0]) <= pd.Timestamp(origin):
        raise ValueError("Forecast dates must be after the origin.")
    horizon_of = {d: i + 1 for i, d in enumerate(dates)}

    df = forecasts[["unique_id", "ds", *w_est]].merge(
        actuals[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left", validate="one_to_one"
    )
    if df["y"].isna().any():
        raise ValueError("Some forecasts have no matching actual.")
    df = df.merge(scales[["unique_id", "mase_scale", "rmsse_scale"]], on="unique_id",
                  how="left", validate="many_to_one")
    if df["mase_scale"].isna().any():
        raise ValueError("Some series have no scale.")

    df["horizon"] = df["ds"].map(horizon_of).astype("int64")
    df["_order"] = df["unique_id"].map(order)
    df = df.sort_values(["_order", "horizon"])

    long = df.melt(
        id_vars=["unique_id", "horizon", "y", "mase_scale", "rmsse_scale"],
        value_vars=list(w_est), var_name="W_est", value_name="yhat",
    )
    long["W_est"] = long["W_est"].map(w_est)
    long = long.rename(columns={"unique_id": "series_id"})
    long["level"] = long["series_id"].map(level_of)
    long["dataset"] = dataset
    long["origin"] = pd.Timestamp(origin)
    long["model"] = model
    if quantiles is None:
        for q in QUANTILE_COLS:
            long[q] = np.nan
    else:
        qdf = quantiles[["unique_id", "ds", "W_est", *QUANTILE_COLS]].copy()
        unknown = set(qdf["W_est"]) - set(w_est.values())
        if unknown or not set(qdf["ds"]) <= set(horizon_of):
            raise ValueError("quantiles has W_est labels or dates that are not in forecasts.")
        qdf["horizon"] = qdf["ds"].map(horizon_of).astype("int64")
        qdf = qdf.rename(columns={"unique_id": "series_id"}).drop(columns="ds")
        long = long.merge(qdf, on=["series_id", "horizon", "W_est"], how="left",
                          validate="one_to_one")
    for c in ["y", "yhat", "mase_scale", "rmsse_scale", *QUANTILE_COLS]:
        long[c] = long[c].astype("float64")
    return long[COLUMNS].reset_index(drop=True)


def append_results(path, new):
    """Append rows to the results table at `path` and return the full table.

    Existing rows with the same (dataset, origin, model, W_est) as any new row
    are replaced, so re-running a script does not duplicate rows.
    """
    if list(new.columns) != COLUMNS:
        raise ValueError(f"Columns must be exactly {COLUMNS}")
    path = Path(path)
    if path.exists():
        old = pd.read_parquet(path)
        new_keys = new[KEY_COLS].drop_duplicates()
        flagged = old.merge(new_keys, on=KEY_COLS, how="left", indicator=True)
        old = old[(flagged["_merge"] == "left_only").to_numpy()]
        full = pd.concat([old, new], ignore_index=True) if len(old) else new.reset_index(drop=True)
    else:
        full = new.reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(path, index=False)
    return full
