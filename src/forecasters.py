"""Base forecasters behind one interface, with a disk cache.

    forecast(model, context_df, h) -> DataFrame[unique_id, ds, yhat, mean, q10..q90]

`yhat` is the median and `mean` the mean forecast; what they are for each kind
of model is described in src/runners.py, which also holds the model registry.
Every base forecast is cached to results/ keyed by (dataset, origin, model),
where origin is the last timestamp in context_df, and is never re-run once
cached (CLAUDE.md).

A model whose registry entry names another conda env is run there in a
subprocess and its forecasts are read back from parquet, so the two envs never
share a process.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.runners import COLUMNS, MODELS, QUANTILE_COLS, QUANTILE_LEVELS, run_cutoffs  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def point_forecast(base, point):
    """The point forecast of each row of a `forecast` frame.

    median: the registered point forecast.
    mean: the `mean` column. For quantile models it is the average of the native
        quantile levels (Deviations log, mean arm), which leaves out the tails
        beyond the lowest and highest level. For sample models it is the sample
        mean. Frames cached before the column existed come from a model with the
        9 levels 0.1..0.9, so the average of q10..q90 is used for them.
    mean9: the average of the 9 levels 0.1..0.9 (Deviations log, mean-arm
        sensitivity). It differs from `mean` only for a model with more native
        levels, such as Chronos-2.
    """
    if point == "median":
        return base["yhat" if "yhat" in base else "q50"].to_numpy(dtype=np.float64)
    if point == "mean":
        if "mean" in base:
            return base["mean"].to_numpy(dtype=np.float64)
        return base[QUANTILE_COLS].to_numpy(dtype=np.float64).mean(axis=1)
    if point == "mean9":
        return base[QUANTILE_COLS].to_numpy(dtype=np.float64).mean(axis=1)
    raise ValueError(f"point must be 'median', 'mean' or 'mean9', got {point!r}.")


def with_mean(frame):
    """Add the `mean` column to a frame cached before the column existed."""
    if "mean" not in frame:
        frame = frame.copy()
        frame.insert(frame.columns.get_loc("q10"), "mean", point_forecast(frame, "mean"))
    return frame


def with_sample_var(frame, samples_file):
    """Add sample_var to the frame of a sample-path model cached before the
    column existed, from its stored sample paths."""
    if "sample_var" in frame:
        return frame
    s = pd.read_parquet(samples_file)
    if list(s["unique_id"]) != list(frame["unique_id"]) or list(s["ds"]) != list(frame["ds"]):
        raise RuntimeError(f"{samples_file} does not line up with the cached forecast.")
    draws = s[[c for c in s.columns if c not in ("unique_id", "ds")]].to_numpy(dtype=np.float64)
    return frame.assign(sample_var=draws.var(axis=1, ddof=1))


def current_env():
    return os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name


def cache_paths(dataset, origin, model, cache_dir=RESULTS):
    key = f"{dataset}_{pd.Timestamp(origin).date()}_{model}"
    cache_dir = Path(cache_dir)
    return cache_dir / f"base_{key}.parquet", cache_dir / f"timing_{key}.parquet"


def samples_path(dataset, origin, model, cache_dir=RESULTS):
    return Path(cache_dir) / f"samples_{dataset}_{pd.Timestamp(origin).date()}_{model}.parquet"


def _run_in_env(env, model, y_df, cutoffs, h, freq, device, keep_samples, trim):
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda is None:
        raise RuntimeError(f"conda not found; it is needed to run {model} in env {env}.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        y_df[["unique_id", "ds", "y"]].to_parquet(tmp / "input.parquet", index=False)
        cmd = [conda, "run", "--no-capture-output", "-n", env, "python", "-m", "src.forecast_worker",
               "--model", model, "--input", str(tmp / "input.parquet"),
               "--output", str(tmp / "output.parquet"), "--info", str(tmp / "info.json"),
               "--cutoffs", ",".join(str(pd.Timestamp(c).date()) for c in cutoffs),
               "--h", str(h), "--freq", freq, "--device", device]
        if keep_samples:
            (tmp / "samples").mkdir()
            cmd += ["--samples-dir", str(tmp / "samples")]
        if trim:
            cmd.append("--trim-leading-zeros")
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{model} failed in env {env}:\n{proc.stderr[-3000:]}")
        forecasts = pd.read_parquet(tmp / "output.parquet")
        info = json.loads((tmp / "info.json").read_text())
        info["samples"] = {pd.Timestamp(p.stem): pd.read_parquet(p)
                           for p in sorted((tmp / "samples").glob("*.parquet"))} if keep_samples else {}
    for col in ["cutoff", "ds"]:
        forecasts[col] = pd.to_datetime(forecasts[col]).astype("datetime64[ns]")
    return forecasts, info


def forecast_cutoffs(model, y_df, cutoffs, h, *, freq=None, device="cpu", keep_samples=False,
                     trim_leading_zeros=False):
    """Forecast h steps from each cutoff, each with the data up to that cutoff
    only. The model is loaded once for all cutoffs.

    trim_leading_zeros: the M5 rule, see src.runners.run_cutoffs.

    Returns (forecasts, info). forecasts has a `cutoff` column followed by
    unique_id, ds, yhat, mean, q10..q90 and any further native quantiles. info
    holds the revision, device, load time and the seconds per cutoff.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}. Known: {sorted(MODELS)}")
    if y_df["y"].isna().any():
        raise ValueError("The data contains NaN in y.")
    if freq is None:
        freq = pd.infer_freq(pd.DatetimeIndex(sorted(y_df["ds"].unique())))
        if freq is None:
            raise ValueError("Could not infer the frequency; pass freq.")

    env = MODELS[model].get("env")
    if env is None or env == current_env():
        forecasts, info = run_cutoffs(model, y_df, cutoffs, h, freq, device, keep_samples,
                                      trim_leading_zeros_=trim_leading_zeros)
    else:
        forecasts, info = _run_in_env(env, model, y_df, cutoffs, h, freq, device, keep_samples,
                                      trim_leading_zeros)
    info["env"] = env or current_env()

    values = forecasts[["yhat", "mean", *QUANTILE_COLS]].to_numpy()
    if not np.isfinite(values).all():
        raise RuntimeError(f"{model}: the forecast contains NaN or inf.")
    return forecasts, info


def forecast(model, context_df, h, *, freq=None, device="cpu", dataset=None, cache_dir=RESULTS,
             trim_leading_zeros=None):
    """Forecast h steps ahead for every series in context_df.

    context_df: long frame with unique_id, ds, y. Each series is passed to the
        model with its full history as context. All series must end at the same
        timestamp (the origin).
    freq: pandas frequency of the series. Inferred from `ds` if None.
    dataset: if given, the forecast is read from / written to the cache in
        `cache_dir`, keyed by (dataset, origin, model). Sample paths of
        sample-path models are written next to it.
    trim_leading_zeros: the M5 rule (src.runners.run_cutoffs). By default it is
        taken from the dataset's entry in src.data.DATASETS.

    Returns a frame with unique_id, ds, yhat (the median), mean and q10..q90,
    followed by any further native quantiles, in the order in which the series
    appear in context_df.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}. Known: {sorted(MODELS)}")

    last = context_df.groupby("unique_id", sort=False)["ds"].max()
    if last.nunique() != 1:
        raise ValueError("All series in context_df must end at the same timestamp.")
    origin = pd.Timestamp(last.iloc[0])

    if dataset is not None:
        base_path, timing_path = cache_paths(dataset, origin, model, cache_dir)
        if base_path.exists():
            cached = with_mean(pd.read_parquet(base_path))
            if MODELS[model]["output"] == "samples":
                cached = with_sample_var(cached, samples_path(dataset, origin, model, cache_dir))
            return cached

    if trim_leading_zeros is None:
        trim_leading_zeros = dataset_rule(dataset)
    forecasts, info = forecast_cutoffs(model, context_df, [origin], h, freq=freq, device=device,
                                       keep_samples=dataset is not None,
                                       trim_leading_zeros=trim_leading_zeros)
    out = forecasts.drop(columns="cutoff")

    if dataset is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        out.to_parquet(base_path, index=False)
        for frame in info["samples"].values():
            frame.to_parquet(samples_path(dataset, origin, model, cache_dir), index=False)
        pd.DataFrame({
            "stage": ["base_forecast"], "seconds": [info["seconds"][0]],
            "load_seconds": [info["load_seconds"]], "device": [device], "env": [info["env"]],
            "repo": [info["repo"]], "revision": [info["revision"]],
            "zero_forecasts": [info["zero_forecasts"][0]], "trim_leading_zeros": [info["trim_leading_zeros"]],
        }).to_parquet(timing_path, index=False)
    return out


def dataset_rule(dataset):
    """Whether a dataset trims leading zeros (DATASETS[...]["trim_leading_zeros"])."""
    if dataset is None:
        return False
    from src.data import DATASETS

    return bool(DATASETS.get(dataset, {}).get("trim_leading_zeros", False))
