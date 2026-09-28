"""Diagnostic: seasonal-naive in-sample MASE scale on TourismLarge.

Holds out the last h months and computes, per series,
    scale = mean_t |y_t - y_{t-m}|  over the training window, m = 12.
Reports how many series have a zero or near-zero scale, by level.

Run from the repo root: python notebooks/01_zero_scale.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_hierarchy  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

M = 12
TOL = 1e-8
K = 5

Y_df, S_df, tags, freq, h = load_hierarchy("TourismLarge")
level_of = {uid: level for level, ids in tags.items() for uid in ids}

wide = Y_df.pivot(index="unique_id", columns="ds", values="y").loc[S_df["unique_id"]]
T = wide.shape[1]


def scale_table(n_holdout):
    train = wide.iloc[:, : T - n_holdout]
    y = train.to_numpy()
    diffs = np.abs(y[:, M:] - y[:, :-M])
    out = pd.DataFrame(
        {
            "level": [level_of[u] for u in train.index],
            "scale": diffs.mean(axis=1),
            "train_mean": y.mean(axis=1),
            "train_zero_share": (y == 0).mean(axis=1),
            "holdout_mean": wide.iloc[:, T - n_holdout : T - n_holdout + h].to_numpy().mean(axis=1),
        },
        index=train.index,
    )
    out["rel_scale"] = out["scale"] / out["train_mean"].replace(0, np.nan)
    return out, train


tab, train = scale_table(h)
print(f"TourismLarge, last {h} months held out")
print(f"training window: {train.columns[0].date()} to {train.columns[-1].date()} ({train.shape[1]} months)")
print(f"scale terms per series: {train.shape[1] - M}")
print()

levels = list(tags.keys())
by_level = tab.groupby("level", sort=False).agg(
    n_series=("scale", "size"),
    n_scale_eq_0=("scale", lambda s: int((s == 0).sum())),
    n_scale_lt_tol=("scale", lambda s: int((s < TOL).sum())),
    min_scale=("scale", "min"),
    median_scale=("scale", "median"),
    min_rel_scale=("rel_scale", "min"),
    n_any_zero_obs=("train_zero_share", lambda s: int((s > 0).sum())),
    max_zero_share=("train_zero_share", "max"),
).loc[levels]
print("Per level:")
print(by_level.to_string(float_format=lambda x: f"{x:.4g}"))
print()
print(f"TOTAL scale == 0: {int((tab.scale == 0).sum())} | scale < {TOL}: {int((tab.scale < TOL).sum())} of {len(tab)}")
print()

print("20 smallest scales:")
print(tab.nsmallest(20, "scale").to_string(float_format=lambda x: f"{x:.4g}"))
print()

print("10 series with the highest share of zero observations in training:")
print(tab.nlargest(10, "train_zero_share").to_string(float_format=lambda x: f"{x:.4g}"))
print()

print("Intermittency: series per level by share of zero observations in training")
intermittency = tab.groupby("level", sort=False).agg(
    n_series=("train_zero_share", "size"),
    n_gt_25=("train_zero_share", lambda s: int((s > 0.25).sum())),
    n_gt_50=("train_zero_share", lambda s: int((s > 0.50).sum())),
    n_gt_75=("train_zero_share", lambda s: int((s > 0.75).sum())),
).loc[levels]
for c in ["n_gt_25", "n_gt_50", "n_gt_75"]:
    intermittency[c.replace("n_", "share_")] = intermittency[c] / intermittency["n_series"]
print(intermittency.to_string(float_format=lambda x: f"{x:.1%}"))
print()

print(f"Same count at each of the {K} rolling origins (origins {h} months apart, ending at the last observation):")
rows = []
for k in range(K):
    n_holdout = h * (k + 1)
    t, tr = scale_table(n_holdout)
    rows.append(
        {
            "origin_k": K - k,
            "train_end": tr.columns[-1].date(),
            "train_len": tr.shape[1],
            "n_scale_eq_0": int((t.scale == 0).sum()),
            "n_scale_lt_tol": int((t.scale < TOL).sum()),
            "min_scale": t.scale.min(),
            "min_scale_id": t.scale.idxmin(),
        }
    )
print(pd.DataFrame(rows).sort_values("origin_k").to_string(index=False, float_format=lambda x: f"{x:.4g}"))
