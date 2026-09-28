"""Base forecasters behind one interface, with a disk cache.

    forecast(model, context_df, h) -> DataFrame[unique_id, ds, yhat, q10..q90]

`yhat` is the median (q50). Every base forecast is cached to results/ keyed by
(dataset, origin, model), where origin is the last timestamp in context_df, and
is never re-run once cached (CLAUDE.md).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANTILE_COLS = [f"q{int(round(q * 100))}" for q in QUANTILE_LEVELS]
COLUMNS = ["unique_id", "ds", "yhat", *QUANTILE_COLS]

_PIPELINES = {}


def _check_openmp_clash():
    """hierarchicalforecast's compiled extension and torch each bundle their own
    OpenMP runtime. On macOS, loading both in one process hangs or segfaults at
    the first model call, so fail with a message instead."""
    if sys.platform == "darwin" and "hierarchicalforecast._lib" in sys.modules:
        raise RuntimeError(
            "hierarchicalforecast is loaded in this process. It cannot share a process with "
            "torch on macOS (two OpenMP runtimes). Run foundation-model forecasts in a process "
            "that does not import hierarchicalforecast."
        )


def _chronos_bolt(repo):
    def run(contexts, h, device):
        _check_openmp_clash()
        import torch
        from chronos import BaseChronosPipeline

        key = (repo, device)
        if key not in _PIPELINES:
            _PIPELINES[key] = BaseChronosPipeline.from_pretrained(
                repo, device_map=device, torch_dtype=torch.float32
            )
        pipeline = _PIPELINES[key]
        out = []
        batch_size = 256
        for i in range(0, len(contexts), batch_size):
            batch = [torch.tensor(c, dtype=torch.float32) for c in contexts[i : i + batch_size]]
            quantiles, _ = pipeline.predict_quantiles(
                batch, prediction_length=h, quantile_levels=QUANTILE_LEVELS
            )
            out.append(quantiles.cpu().numpy().astype(np.float64))
        return np.concatenate(out, axis=0)  # (n_series, h, 9)

    return run


# model name -> (function returning quantiles of shape (n_series, h, 9), Hugging Face repo)
MODELS = {
    "chronos_bolt_small": (_chronos_bolt("amazon/chronos-bolt-small"), "amazon/chronos-bolt-small"),
}


def cache_paths(dataset, origin, model, cache_dir=RESULTS):
    key = f"{dataset}_{pd.Timestamp(origin).date()}_{model}"
    cache_dir = Path(cache_dir)
    return cache_dir / f"base_{key}.parquet", cache_dir / f"timing_{key}.parquet"


def _model_revision(repo):
    """Commit hash of the Hugging Face snapshot in the local cache, if found."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo, local_files_only=True)).name
    except Exception:
        return "unknown"


def forecast(model, context_df, h, *, freq=None, device="cpu", dataset=None, cache_dir=RESULTS):
    """Forecast h steps ahead for every series in context_df.

    context_df: long frame with unique_id, ds, y. Each series is passed to the
        model with its full history as context. All series must end at the same
        timestamp (the origin).
    freq: pandas frequency of the series. Inferred from `ds` if None.
    dataset: if given, the forecast is read from / written to the cache in
        `cache_dir`, keyed by (dataset, origin, model).

    Returns a frame with unique_id, ds, yhat (the median) and q10..q90, in the
    order in which the series appear in context_df.
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
            return pd.read_parquet(base_path)

    if context_df["y"].isna().any():
        raise ValueError("context_df contains NaN in y.")
    context_df = context_df.sort_values(["unique_id", "ds"], kind="stable")
    ids = list(last.index)
    groups = context_df.groupby("unique_id", sort=False)["y"]
    contexts = [groups.get_group(uid).to_numpy(dtype=np.float64) for uid in ids]

    if freq is None:
        freq = pd.infer_freq(pd.DatetimeIndex(sorted(context_df["ds"].unique())))
        if freq is None:
            raise ValueError("Could not infer the frequency; pass freq.")
    future = pd.date_range(origin, periods=h + 1, freq=freq)[1:]

    run, repo = MODELS[model]
    t0 = time.perf_counter()
    quantiles = run(contexts, h, device)
    seconds = time.perf_counter() - t0
    if quantiles.shape != (len(ids), h, len(QUANTILE_LEVELS)):
        raise RuntimeError(f"Unexpected forecast shape {quantiles.shape}.")
    if not np.isfinite(quantiles).all():
        raise RuntimeError("Forecast contains NaN or inf.")

    out = pd.DataFrame({
        "unique_id": np.repeat(np.asarray(ids, dtype=object), h),
        "ds": np.tile(future, len(ids)),
        "yhat": quantiles[:, :, QUANTILE_LEVELS.index(0.5)].reshape(-1),
    })
    for j, col in enumerate(QUANTILE_COLS):
        out[col] = quantiles[:, :, j].reshape(-1)
    out = out[COLUMNS]

    if dataset is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        out.to_parquet(base_path, index=False)
        pd.DataFrame({
            "stage": ["base_forecast"], "seconds": [seconds], "device": [device],
            "repo": [repo], "revision": [_model_revision(repo)],
        }).to_parquet(timing_path, index=False)
    return out
