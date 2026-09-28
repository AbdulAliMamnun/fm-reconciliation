"""Point-forecast metrics (PREREG §6 and the Deviations log).

Scales are always computed from the training data only. Per-series metrics
average over the last axis (the horizon), so inputs can be 1-D (one series)
or 2-D (series x horizon).
"""
import numpy as np
import pandas as pd

SCALE_TOL = 1e-8


def _seasonal_diffs(y_train, m):
    y_train = np.asarray(y_train, dtype=float)
    if y_train.shape[-1] <= m:
        raise ValueError(
            f"Need more than m={m} training observations, got {y_train.shape[-1]}."
        )
    return y_train[..., m:] - y_train[..., :-m]


def scale_mase(y_train, m):
    """Mean absolute seasonal-naive error in-sample: mean |y_t - y_{t-m}|."""
    return np.mean(np.abs(_seasonal_diffs(y_train, m)), axis=-1)


def scale_rmsse(y_train, m):
    """Mean squared seasonal-naive error in-sample: mean (y_t - y_{t-m})^2.

    This is the squared scale; `rmsse` takes the square root of the ratio.
    """
    return np.mean(_seasonal_diffs(y_train, m) ** 2, axis=-1)


def mase(y, yhat, scale):
    """mean_h |y - yhat| / scale."""
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    return np.mean(np.abs(y - yhat), axis=-1) / scale


def rmsse(y, yhat, scale_sq):
    """sqrt( mean_h (y - yhat)^2 / scale_sq )."""
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    return np.sqrt(np.mean((y - yhat) ** 2, axis=-1) / scale_sq)


def level_mean(metric_df, tags, value_cols=None, by=None, scale_col="scale",
               id_col="unique_id", tol=SCALE_TOL):
    """Arithmetic mean over series within each level.

    Series with `scale_col` < tol are excluded from every value column and
    counted in `n_excluded`.

    metric_df: one row per series (per `by` group, if given), with columns
        `id_col`, `scale_col` (the seasonal-naive MASE scale) and the metric
        columns.
    tags: dict mapping level name -> array of series ids. Levels are returned
        in this order.
    value_cols: metric columns to average. Default: every column except
        `id_col`, `scale_col` and `by`.
    by: optional list of columns to group by in addition to the level
        (e.g. model, W_est, origin).

    Returns a frame with columns [*by, level, n_series, n_excluded, *value_cols],
    where n_series is the number of series that entered the mean.
    """
    by = list(by) if by else []
    if value_cols is None:
        value_cols = [c for c in metric_df.columns if c not in [id_col, scale_col, *by]]

    dup_cols = [*by, id_col]
    if metric_df.duplicated(dup_cols).any():
        raise ValueError(f"metric_df has duplicate rows for {dup_cols}.")

    out = []
    for level, ids in tags.items():
        ids = list(ids)
        sub = metric_df[metric_df[id_col].isin(ids)]
        groups = sub.groupby(by, sort=False) if by else [((), sub)]
        for key, g in groups:
            missing = set(ids) - set(g[id_col])
            if missing:
                raise ValueError(
                    f"Level {level!r}: {len(missing)} series missing from metric_df, "
                    f"e.g. {sorted(missing)[:3]}."
                )
            keep = g[scale_col] >= tol
            row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
            row["level"] = level
            row["n_series"] = int(keep.sum())
            row["n_excluded"] = int((~keep).sum())
            for c in value_cols:
                row[c] = g.loc[keep, c].mean() if keep.any() else np.nan
            out.append(row)

    return pd.DataFrame(out, columns=[*by, "level", "n_series", "n_excluded", *value_cols])
