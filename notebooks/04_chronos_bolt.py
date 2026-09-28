"""Week 2, step 1: Chronos-Bolt-small on TourismLarge, 5 origins, two point-forecast arms.

Base forecasts (q10..q90) are cached per origin in results/ and reconciled with
the numpy reconciler using OLS, WLS-struct and WLS-pv. WLS-pv uses each
horizon's own predictive variances. No backtest residuals here.

Arms (PREREG Deviations log, mean-forecast arm):
  median  the registered point forecast, q50. Primary.
  mean    the average of the 9 quantiles. EXPLORATORY for Chronos-Bolt x
          TourismLarge: it was added after the median results were seen.
The same W is used in both arms; only the point forecast that is reconciled
and scored differs.

Everything is appended to results/results.parquet and all tables below are
computed from that table. The AutoETS rows written by notebooks/02_gate_ets.py
are used as reference columns.

Do not import hierarchicalforecast here: it cannot share a process with torch
on macOS (see src/forecasters.py).

Run from the repo root: python notebooks/04_chronos_bolt.py [--per-origin]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import DATASETS, load_hierarchy, rolling_origins  # noqa: E402
from src.forecasters import QUANTILE_COLS, cache_paths, forecast, point_forecast  # noqa: E402
from src.metrics import level_mean, scale_mase, scale_rmsse, series_metrics  # noqa: E402
from src.reconcile import (  # noqa: E402
    W_ols, W_pv, W_struct, predictive_variance, reconcile, reconcile_by_horizon,
)
from src.results import append_results, build_results  # noqa: E402

DATASET = "TourismLarge"
MODEL = "chronos_bolt_small"
DEVICE = "cpu"
K = 5
COHERENCE_TOL = 1e-8
TABLE = ROOT / "results" / "results.parquet"

POINTS = ["median", "mean"]
W_ESTS = ["base", "ols", "wls_struct", "wls_pv"]
W_LABEL = {"base": "Base", "ols": "OLS", "wls_struct": "WLS-struct", "wls_pv": "WLS-pv"}
# (model, point, W_est) -> column label, in table order
COLS = {(MODEL, p, w): f"{p}:{W_LABEL[w]}" for p in POINTS for w in W_ESTS}
COLS[("AutoETS", "mean", "base")] = "ETS:Base"
COLS[("AutoETS", "mean", "mint_shrink")] = "ETS:MinT-shrink"
LABELS = list(COLS.values())
ARM_LABELS = {p: [f"{p}:{W_LABEL[w]}" for w in W_ESTS] for p in POINTS}
EXPLORATORY = "EXPLORATORY (mean arm; post hoc for Chronos-Bolt x TourismLarge)"


def mark_best(tab, counts=None):
    """+ marks the best column within each Chronos-Bolt arm, * the best of all."""
    out = tab.map(lambda x: f"{x:.4f}")
    best = tab.idxmin(axis=1)
    for labels in ARM_LABELS.values():
        best_arm = tab[labels].idxmin(axis=1)
        for idx in tab.index:
            out.loc[idx, best_arm[idx]] += "+"
    for idx in tab.index:
        out.loc[idx, best[idx]] += "*"
    if counts is not None:
        out.insert(0, "excl", counts["n_excluded"])
        out.insert(0, "n", counts["n_series"])
    return out


def main():
    per_origin = "--per-origin" in sys.argv
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    t_start = time.perf_counter()

    Y_df, S_df, tags, freq, h = load_hierarchy(DATASET)
    m = DATASETS[DATASET]["m"]
    ids = S_df["unique_id"].to_numpy()
    bottom_ids = [c for c in S_df.columns if c != "unique_id"]
    S = S_df[bottom_ids].to_numpy(dtype=np.float64)
    n, nb = S.shape
    wide = Y_df.pivot(index="unique_id", columns="ds", values="y").loc[ids]

    print(f"{DATASET}: {n} series, {K} origins, h={h}, model {MODEL}, device {DEVICE}")
    print(f"arms: median (registered, primary) and mean = average of the 9 quantiles. {EXPLORATORY}")
    new_rows, run_rows, diag_rows = [], [], []
    for split in rolling_origins(Y_df, h, K):
        origin, train, test = split["origin"], split["train"], split["test"]
        base_path, timing_path = cache_paths(DATASET, origin, MODEL)
        cached = base_path.exists()
        base = forecast(MODEL, train, h, freq=freq, device=DEVICE, dataset=DATASET)
        timing = pd.read_parquet(timing_path).iloc[0]
        dates = np.sort(base["ds"].unique())

        def to_wide(values):
            return (base.assign(_v=values).pivot(index="unique_id", columns="ds", values="_v")
                    .loc[ids].to_numpy())

        q = np.stack([to_wide(base[c].to_numpy()) for c in QUANTILE_COLS], axis=-1)  # (n, h, 9)
        v = predictive_variance(q)
        width = q[..., -1] - q[..., 0]
        Ws_pv = W_pv(q)

        y_train = wide.loc[:, :origin].to_numpy()
        scales = pd.DataFrame({"unique_id": ids, "mase_scale": scale_mase(y_train, m),
                               "rmsse_scale": scale_rmsse(y_train, m)})
        quant = base[["unique_id", "ds", *QUANTILE_COLS]].assign(W_est="base")

        seconds = {}
        for point in POINTS:
            y_hat = to_wide(point_forecast(base, point))                  # (n, h)
            rec = {"base": y_hat}
            t0 = time.perf_counter()
            rec["ols"] = reconcile(y_hat, S, W_ols(n))
            rec["wls_struct"] = reconcile(y_hat, S, W_struct(S))
            rec["wls_pv"] = reconcile_by_horizon(y_hat, S, Ws_pv)
            seconds[point] = time.perf_counter() - t0

            fc = pd.DataFrame({"unique_id": np.repeat(ids, h), "ds": np.tile(dates, n)})
            for w in W_ESTS:
                fc[w] = rec[w].reshape(-1)
            new_rows.append(build_results(DATASET, origin, MODEL, fc, test, tags, scales,
                                          {w: w for w in W_ESTS}, point=point, quantiles=quant))

        med, mean = to_wide(point_forecast(base, "median")), to_wide(point_forecast(base, "mean"))
        run_rows.append({
            "origin": origin.date(), "train_len": y_train.shape[1],
            "base": "cache" if cached else "computed", "device": timing["device"],
            "base_forecast_s": timing["seconds"],
            **{f"reconcile_{p}_s": s for p, s in seconds.items()},
            "revision": timing["revision"][:10],
        })
        diag_rows.append({
            "origin": origin.date(),
            "quantile_crossings": int((np.diff(q, axis=-1) < 0).any(axis=-1).sum()),
            "q90<=q10": int((width <= 0).sum()),
            "min_width": width.min(),
            "max_v/min_v": v.max() / v.min(),
            "neg_median": int((med < 0).sum()),
            "neg_mean": int((mean < 0).sum()),
            "mean>median": int((mean > med).sum()),
            "cells": med.size,
        })
        print(f"  origin {origin.date()}: train {y_train.shape[1]} months "
              f"[base: {'cache' if cached else 'computed'}]")

    append_results(TABLE, pd.concat(new_rows, ignore_index=True))

    # ---- everything below reads only the results table ----
    t0 = time.perf_counter()
    res_all = pd.read_parquet(TABLE)
    res_all = res_all[res_all["dataset"] == DATASET]
    keys = pd.Series(list(zip(res_all["model"], res_all["point"], res_all["W_est"])), index=res_all.index)
    res = res_all[keys.isin(list(COLS))].copy()
    missing = set(COLS) - set(keys)
    if missing:
        raise SystemExit(f"Missing from the results table: {missing}. Run notebooks/02_gate_ets.py.")
    res["col"] = [COLS[k] for k in zip(res["model"], res["point"], res["W_est"])]
    origins = sorted(res["origin"].unique())
    cb = res[res.model == MODEL]
    print(f"\nresults table: {len(res_all)} rows for {DATASET}; {len(cb)} for {MODEL} "
          f"({cb.origin.nunique()} origins x {cb.point.nunique()} arms x {cb.W_est.nunique()} W_est x "
          f"{cb.series_id.nunique()} series x {cb.horizon.nunique()} horizons); "
          f"NaN yhat: {int(cb.yhat.isna().sum())}")

    print("\nBase forecast diagnostics (per origin, over series x horizons):")
    print(pd.DataFrame(diag_rows).to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    coh_rows = []
    for (origin, col), g in cb.groupby(["origin", "col"], sort=False):
        yhat = g.pivot(index="series_id", columns="horizon", values="yhat")
        coh_rows.append({
            "origin": pd.Timestamp(origin).date(), "col": col,
            "max_gap": np.abs(S @ yhat.loc[bottom_ids].to_numpy() - yhat.loc[ids].to_numpy()).max(),
            "n_negative": int((yhat.to_numpy() < 0).sum()),
        })
    coh = pd.DataFrame(coh_rows)
    cb_labels = ARM_LABELS["median"] + ARM_LABELS["mean"]
    print("\nCoherence, max |S b - y| by origin:")
    print(coh.pivot(index="origin", columns="col", values="max_gap")[cb_labels]
          .to_string(float_format=lambda x: f"{x:.2e}"))
    worst = coh[~coh.col.str.endswith(":Base")]["max_gap"].max()
    print(f"all reconciled forecasts coherent (tolerance {COHERENCE_TOL:g}): "
          f"{'YES' if worst <= COHERENCE_TOL else 'NO'} (worst {worst:.3e})")
    print("\nNegative point forecasts by origin (count over series x horizons):")
    print(coh.pivot(index="origin", columns="col", values="n_negative")[cb_labels].to_string())

    sm = series_metrics(res.drop(columns="col"))
    sm["col"] = [COLS[k] for k in zip(sm["model"], sm["point"], sm["W_est"])]
    level_order = list(tags)
    tags_tab = {**{k: np.asarray(val) for k, val in tags.items()}, "All series": ids}
    levels = list(tags_tab)
    lm = level_mean(sm, tags_tab, value_cols=["mase", "rmsse"], by=["origin", "col"],
                    scale_col="mase_scale", id_col="series_id")
    counts = (lm[lm.col == "median:Base"].groupby("level")
              .agg(n_series=("n_series", "first"), n_excluded=("n_excluded", "sum")).loc[levels])
    avg = lm.groupby(["level", "col"], sort=False)[["mase", "rmsse"]].mean().reset_index()

    def avg_table(value):
        return avg.pivot(index="level", columns="col", values=value).loc[levels, LABELS]

    mase_tab, rmsse_tab = avg_table("mase"), avg_table("rmsse")
    legend = "(+ = best within its Chronos-Bolt arm, * = best of all columns)"
    print(f"\nSeries excluded by the scale < 1e-8 rule, summed over origins: "
          f"{int(counts.loc['All series', 'n_excluded'])}")
    print(f"\nColumns 'mean:*' are {EXPLORATORY}")
    print(f"\nMean MASE by level, averaged over {len(origins)} origins   {legend}")
    print(mark_best(mase_tab, counts).to_string())
    print(f"\nMean RMSSE by level, averaged over {len(origins)} origins   {legend}")
    print(mark_best(rmsse_tab, counts).to_string())

    for name, tab in [("MASE", mase_tab), ("RMSSE", rmsse_tab)]:
        gain = pd.DataFrame()
        for p in POINTS:
            for w in W_ESTS[1:]:
                gain[f"{p}:{W_LABEL[w]}"] = (tab[f"{p}:{W_LABEL[w]}"] - tab[f"{p}:Base"]) / tab[f"{p}:Base"]
        gain["mean:Base vs median:Base"] = (tab["mean:Base"] - tab["median:Base"]) / tab["median:Base"]
        print(f"\nRelative change in {name}, negative = better. Each reconciled column is compared "
              "with the Base of its own arm.")
        print((100 * gain).to_string(float_format=lambda x: f"{x:+.2f}%"))

    top, bottom_level = levels[0], level_order[-1]
    lm_top = lm[lm.level == top].assign(origin=lambda d: d["origin"].dt.date)
    for value in ["mase", "rmsse"]:
        tab = lm_top.pivot(index="origin", columns="col", values=value)[LABELS]
        tab.loc["mean over origins"] = tab.mean()
        print(f"\n{top} {value.upper()} by origin   {legend}")
        print(mark_best(tab).to_string())

    bias = sm[sm.level == top].assign(origin=lambda d: d["origin"].dt.date)
    bias_tab = bias.pivot(index="origin", columns="col", values="bias")[LABELS]
    mean_y = (res[(res.level == top) & (res.col == "median:Base")]
              .groupby("origin")["y"].mean().rename(lambda o: pd.Timestamp(o).date()))
    out = bias_tab.copy()
    out.insert(0, "mean_y", mean_y)
    print(f"\n{top} bias by origin: mean over the 12 horizons of (yhat - y)")
    print(out.to_string(float_format=lambda x: f"{x:.0f}"))
    print(f"\n{top} bias as % of the mean actual")
    print((100 * bias_tab.div(mean_y, axis=0)).to_string(float_format=lambda x: f"{x:+.2f}%"))

    # ---- descriptive diagnostics (exploratory, not a registered test) ----
    base_rows = res[res.W_est == "base"].assign(e=lambda d: d["yhat"] - d["y"])
    add = pd.DataFrame({"actual": mean_y})
    for col in ["median:Base", "mean:Base", "ETS:Base"]:
        sub = base_rows[base_rows.col == col]
        add[f"{col} Country"] = (sub[sub.level == top].groupby("origin")["yhat"].mean()
                                 .rename(lambda o: pd.Timestamp(o).date()))
        add[f"{col} sum of bottom"] = (sub[sub.level == bottom_level].groupby("origin")["yhat"].sum()
                                       .rename(lambda o: pd.Timestamp(o).date()) / h)
    print("\nExploratory: do the bottom base forecasts add up to the Country forecast? "
          "Monthly averages over the 12 horizons")
    print(add.to_string(float_format=lambda x: f"{x:.0f}"))
    print("as % difference from the actual")
    print((100 * add.drop(columns="actual").sub(add["actual"], axis=0).div(add["actual"], axis=0))
          .to_string(float_format=lambda x: f"{x:+.1f}%"))

    sb = sm[sm.W_est == "base"].assign(sbias=lambda d: d["bias"] / d["mase_scale"])
    sbt = (sb.groupby(["level", "col"], sort=False)["sbias"].mean().unstack("col")
           .loc[level_order, ["median:Base", "mean:Base", "ETS:Base"]])
    print("\nExploratory: mean scaled bias of the base forecasts, (yhat - y) / mase_scale, by level")
    print(sbt.to_string(float_format=lambda x: f"{x:+.3f}"))

    cbb = res[res.col == "median:Base"].copy()
    cbb["v"] = predictive_variance(cbb[QUANTILE_COLS].to_numpy())
    cbb["se"] = (cbb["yhat"] - cbb["y"]) ** 2
    g = cbb.groupby("level", sort=False)
    diag = pd.DataFrame({"rms_pred_sd": np.sqrt(g["v"].mean()),
                         "rmse_median_base": np.sqrt(g["se"].mean())}).loc[level_order]
    diag["ratio"] = diag["rms_pred_sd"] / diag["rmse_median_base"]
    diag["share_of_total_weight_1/v"] = ((1 / cbb["v"]).groupby(cbb["level"], sort=False).sum()
                                         / (1 / cbb["v"]).sum())
    print("\nExploratory: predictive sd (from q10, q90) against the realised RMSE of the median "
          "base forecast, pooled over origins and horizons")
    print(diag.to_string(float_format=lambda x: f"{x:.3f}"))

    if per_origin:
        for origin in origins:
            sub = lm[lm.origin == origin]
            for value in ["mase", "rmsse"]:
                tab = sub.pivot(index="level", columns="col", values=value).loc[levels, LABELS]
                print(f"\nOrigin {pd.Timestamp(origin).date()}: mean {value.upper()} by level   {legend}")
                print(mark_best(tab).to_string())

    t_metrics = time.perf_counter() - t0
    runs = pd.DataFrame(run_rows)
    print("\nRuntime in seconds (base forecast time was measured when the cache was written):")
    print(runs.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"  total base forecast {runs['base_forecast_s'].sum():.2f}s; reconciliation "
          f"{runs[[f'reconcile_{p}_s' for p in POINTS]].sum().sum():.2f}s; metrics {t_metrics:.2f}s; "
          f"this run wall clock {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
