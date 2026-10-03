"""Backtest residuals for models that have no in-sample residuals (PREREG §5 and
the Deviations log, backtest design).

For an outer origin t_o, the model re-forecasts h steps ahead from n_inner inner
origins inside the training data:

    inner origins   t_o - h - (n_inner - 1), ..., t_o - h      (1 period apart)
    targets         inner origin + 1, ..., inner origin + h    (all <= t_o)

Each inner forecast is given the history up to its inner origin and nothing
after it. No observation after t_o is read, so the test window is never used.
The result is n_inner h-step residuals per series and horizon.

The inner forecasts (quantiles) are cached once per (dataset, origin, model).
The residuals are cached per (dataset, origin, model, point). While the inner
forecasts of an origin are being made, every finished call is checkpointed, so
an interrupted run continues where it stopped.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasters import (
    MODELS, QUANTILE_COLS, RESULTS, current_env, dataset_rule, forecast_cutoffs, point_forecast,
    with_mean,
)

INNER_COLUMNS = ["unique_id", "inner_origin", "ds", "horizon", "y", "yhat", "mean", *QUANTILE_COLS]
RESIDUAL_COLUMNS = ["unique_id", "inner_origin", "ds", "horizon", "y", "yhat", "resid"]


def backtest_paths(dataset, origin, model, cache_dir=RESULTS):
    key = f"{dataset}_{pd.Timestamp(origin).date()}_{model}"
    cache_dir = Path(cache_dir)
    return {
        "inner": cache_dir / f"backtest_{key}.parquet",
        "parts": cache_dir / "partial" / f"backtest_{key}",
        "timing": cache_dir / f"backtest_timing_{key}.parquet",
        "residuals": lambda point: cache_dir / f"resid_{key}_{point}.parquet",
    }


def inner_origins(dates, h, n_inner):
    """The inner origins for training dates `dates` (sorted, ending at t_o)."""
    dates = pd.DatetimeIndex(dates)
    T = len(dates)
    first = T - 1 - h - (n_inner - 1)      # position of the earliest inner origin
    if first < 1:
        raise ValueError(
            f"{T} training periods are too few for {n_inner} inner origins with h={h}."
        )
    return dates[first: T - h]


def inner_forecasts(model, y_train, h, n_inner=24, *, freq=None, device="cpu", dataset=None,
                    cache_dir=RESULTS, trim_leading_zeros=None):
    """Quantile forecasts from every inner origin, with the actuals.

    y_train: long frame with unique_id, ds, y, every series ending at the outer
        origin t_o.
    Returns a frame with unique_id, inner_origin, ds, horizon, y, q10..q90.
    """
    dates = pd.DatetimeIndex(sorted(y_train["ds"].unique()))
    outer = dates[-1]
    if y_train.groupby("unique_id", sort=False)["ds"].max().nunique() != 1:
        raise ValueError("All series in y_train must end at the same timestamp.")

    if dataset is not None:
        paths = backtest_paths(dataset, outer, model, cache_dir)
        if paths["inner"].exists():
            cached = pd.read_parquet(paths["inner"])
            if "yhat" not in cached:                 # written before the column existed
                cached.insert(cached.columns.get_loc("q10"), "yhat", cached["q50"])
            return with_mean(cached)

    cutoffs = inner_origins(dates, h, n_inner)
    parts = paths["parts"] if dataset is not None else None
    if trim_leading_zeros is None:
        trim_leading_zeros = dataset_rule(dataset)
    out, seconds, info = _inner_calls(model, y_train, cutoffs, h, freq, device, parts, trim_leading_zeros)
    out = out.rename(columns={"cutoff": "inner_origin"})
    out = out.merge(y_train[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left",
                    validate="many_to_one")
    out["horizon"] = out.groupby(["inner_origin", "unique_id"], sort=False).cumcount() + 1
    extra = [c for c in out.columns if c not in INNER_COLUMNS]
    out = out[INNER_COLUMNS + extra]

    if out["y"].isna().any() or out["ds"].max() > outer:
        raise RuntimeError("An inner forecast target lies after the outer origin.")

    if dataset is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        out.to_parquet(paths["inner"], index=False)
        pd.DataFrame({
            "stage": ["backtest_forecasts"], "seconds": [float(np.sum(seconds))],
            "forecast_calls": [len(seconds)], "series": [out["unique_id"].nunique()],
            "device": [device], "load_seconds": [info["load_seconds"]], "env": [info["env"]],
            "revision": [info["revision"]],
        }).to_parquet(paths["timing"], index=False)
        if parts.exists():                       # the checkpoints are no longer needed
            shutil.rmtree(parts)
    return out


def _inner_calls(model, y_train, cutoffs, h, freq, device, parts, trim):
    """Forecast from every cutoff, keeping a checkpoint per finished call.

    With `parts` given, each call is written to `parts` as soon as it finishes
    and calls found there are not run again, so an interrupted run loses at most
    the call that was in progress. A model that runs in another conda env gets
    all its missing calls in one go, because each go loads the model anew.
    """
    done = {}
    if parts is not None:
        parts.mkdir(parents=True, exist_ok=True)
        for c in cutoffs:
            meta = parts / f"{c.date()}.json"
            if meta.exists():                    # written last, so the frame is complete
                done[c] = (pd.read_parquet(parts / f"{c.date()}.parquet"), json.loads(meta.read_text()))
    todo = [c for c in cutoffs if c not in done]
    in_process = MODELS[model].get("env") in (None, current_env())
    groups = [[c] for c in todo] if in_process else ([todo] if todo else [])
    for group in groups:
        f, info = forecast_cutoffs(model, y_train, group, h, freq=freq, device=device,
                                   trim_leading_zeros=trim)
        for c, sec in zip(group, info["seconds"]):
            frame = f[f["cutoff"] == c].reset_index(drop=True)
            meta = {"seconds": sec, "load_seconds": info["load_seconds"], "env": info["env"],
                    "revision": info["revision"]}
            if parts is not None:
                frame.to_parquet(parts / f"{c.date()}.parquet", index=False)
                (parts / f"{c.date()}.json").write_text(json.dumps(meta))
            done[c] = (frame, meta)
    out = pd.concat([done[c][0] for c in cutoffs], ignore_index=True)
    out["cutoff"] = pd.to_datetime(out["cutoff"]).astype("datetime64[ns]")
    metas = [done[c][1] for c in cutoffs]
    info = {"load_seconds": max(m["load_seconds"] for m in metas), "env": metas[0]["env"],
            "revision": metas[0]["revision"]}
    return out, [m["seconds"] for m in metas], info


def backtest_residuals(model, y_train, h, n_inner=24, *, point="median", freq=None, device="cpu",
                       dataset=None, cache_dir=RESULTS, trim_leading_zeros=None):
    """h-step backtest residuals, resid = y - point forecast.

    point: "median" or "mean" (see src.forecasters.point_forecast).
    Returns a frame with unique_id, inner_origin, ds, horizon, y, yhat, resid:
    n_inner rows per series and horizon, every `ds` at or before the outer origin.
    """
    outer = pd.Timestamp(y_train["ds"].max())
    if dataset is not None:
        path = backtest_paths(dataset, outer, model, cache_dir)["residuals"](point)
        if path.exists():
            return pd.read_parquet(path)

    inner = inner_forecasts(model, y_train, h, n_inner, freq=freq, device=device, dataset=dataset,
                            cache_dir=cache_dir, trim_leading_zeros=trim_leading_zeros)
    out = inner[["unique_id", "inner_origin", "ds", "horizon", "y"]].copy()
    out["yhat"] = point_forecast(inner, point)
    out["resid"] = out["y"] - out["yhat"]
    out = out[RESIDUAL_COLUMNS]
    if out["ds"].max() > outer:
        raise RuntimeError("A residual target lies after the outer origin.")
    if dataset is not None:
        out.to_parquet(path, index=False)
    return out


def residual_array(residuals, ids):
    """Residuals as an array (n_series, n_inner, h): series in the order of
    `ids`, inner origins earliest first, horizons 1..h."""
    inner = np.sort(residuals["inner_origin"].unique())
    h = int(residuals["horizon"].max())
    wide = residuals.pivot(index="unique_id", columns=["inner_origin", "horizon"], values="resid")
    cols = pd.MultiIndex.from_product([inner, range(1, h + 1)])
    arr = wide.reindex(index=list(ids), columns=cols).to_numpy(dtype=np.float64)
    if np.isnan(arr).any():
        raise ValueError("Residuals are missing for some series, inner origins or horizons.")
    return arr.reshape(len(ids), len(inner), h)
