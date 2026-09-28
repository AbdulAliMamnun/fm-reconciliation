"""Week 2, step 1: Chronos-Bolt-small on TourismLarge, 5 origins.

Base forecasts (median and q10..q90) are cached per origin in results/ and
reconciled with the numpy reconciler using OLS, WLS-struct and WLS-pv. WLS-pv
uses each horizon's own predictive variances. No backtest residuals here.
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
from src.forecasters import QUANTILE_COLS, cache_paths, forecast  # noqa: E402
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

W_ESTS = ["base", "ols", "wls_struct", "wls_pv"]
# (model, W_est) -> column label, in table order
COLS = {
    (MODEL, "base"): "CB Base",
    (MODEL, "ols"): "CB OLS",
    (MODEL, "wls_struct"): "CB WLS-struct",
    (MODEL, "wls_pv"): "CB WLS-pv",
    ("AutoETS", "base"): "ETS Base",
    ("AutoETS", "mint_shrink"): "ETS MinT-shrink",
}
LABELS = list(COLS.values())
CB_LABELS = LABELS[:4]


def mark_best(tab, counts=None):
    best = tab.idxmin(axis=1)
    best_cb = tab[CB_LABELS].idxmin(axis=1)
    out = tab.map(lambda x: f"{x:.4f}")
    for idx in tab.index:
        out.loc[idx, best_cb[idx]] += "+"
        out.loc[idx, best[idx]] += "*"
    if counts is not None:
        out.insert(0, "excl", counts["n_excluded"])
        out.insert(0, "n", counts["n_series"])
    return out


def main():
    per_origin = "--per-origin" in sys.argv
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t_start = time.perf_counter()

    Y_df, S_df, tags, freq, h = load_hierarchy(DATASET)
    m = DATASETS[DATASET]["m"]
    ids = S_df["unique_id"].to_numpy()
    bottom_ids = [c for c in S_df.columns if c != "unique_id"]
    S = S_df[bottom_ids].to_numpy(dtype=np.float64)
    n, nb = S.shape
    wide = Y_df.pivot(index="unique_id", columns="ds", values="y").loc[ids]

    print(f"{DATASET}: {n} series, {K} origins, h={h}, model {MODEL}, device {DEVICE}")
    new_rows, run_rows, diag_rows = [], [], []
    for split in rolling_origins(Y_df, h, K):
        origin, train, test = split["origin"], split["train"], split["test"]
        base_path, timing_path = cache_paths(DATASET, origin, MODEL)
        cached = base_path.exists()
        base = forecast(MODEL, train, h, freq=freq, device=DEVICE, dataset=DATASET)
        timing = pd.read_parquet(timing_path).iloc[0]

        def to_wide(col):
            return base.pivot(index="unique_id", columns="ds", values=col).loc[ids].to_numpy()

        y_hat = to_wide("yhat")                                        # (n, h)
        q = np.stack([to_wide(c) for c in QUANTILE_COLS], axis=-1)     # (n, h, 9)
        v = predictive_variance(q)
        width = q[..., -1] - q[..., 0]

        rec, seconds = {"base": y_hat}, {}
        t0 = time.perf_counter()
        rec["ols"] = reconcile(y_hat, S, W_ols(n))
        seconds["ols"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        rec["wls_struct"] = reconcile(y_hat, S, W_struct(S))
        seconds["wls_struct"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        rec["wls_pv"] = reconcile_by_horizon(y_hat, S, W_pv(q))
        seconds["wls_pv"] = time.perf_counter() - t0

        dates = np.sort(base["ds"].unique())
        fc = pd.DataFrame({"unique_id": np.repeat(ids, h), "ds": np.tile(dates, n)})
        for w in W_ESTS:
            fc[w] = rec[w].reshape(-1)
        quant = base[["unique_id", "ds", *QUANTILE_COLS]].assign(W_est="base")

        y_train = wide.loc[:, :origin].to_numpy()
        scales = pd.DataFrame({"unique_id": ids, "mase_scale": scale_mase(y_train, m),
                               "rmsse_scale": scale_rmsse(y_train, m)})
        new_rows.append(build_results(DATASET, origin, MODEL, fc, test, tags, scales,
                                      {w: w for w in W_ESTS}, quantiles=quant))

        run_rows.append({
            "origin": origin.date(), "train_len": y_train.shape[1],
            "base": "cache" if cached else "computed", "device": timing["device"],
            "base_forecast_s": timing["seconds"], **{f"{w}_s": s for w, s in seconds.items()},
            "revision": timing["revision"][:10],
        })
        diag_rows.append({
            "origin": origin.date(),
            "quantile_crossings": int((np.diff(q, axis=-1) < 0).any(axis=-1).sum()),
            "q90<=q10": int((width <= 0).sum()),
            "min_width": width.min(),
            "min_v": v.min(), "max_v": v.max(),
            "max_v/min_v": v.max() / v.min(),
            "neg_median": int((y_hat < 0).sum()),
            "neg_q10": int((q[..., 0] < 0).sum()),
            "cells": y_hat.size,
        })
        print(f"  origin {origin.date()}: train {y_train.shape[1]} months "
              f"[base: {'cache' if cached else 'computed'}]")

    append_results(TABLE, pd.concat(new_rows, ignore_index=True))

    # ---- everything below reads only the results table ----
    t0 = time.perf_counter()
    res_all = pd.read_parquet(TABLE)
    res_all = res_all[res_all["dataset"] == DATASET]
    keep = pd.Series(list(zip(res_all["model"], res_all["W_est"])), index=res_all.index).isin(list(COLS))
    res = res_all[keep]
    missing = set(COLS) - set(zip(res["model"], res["W_est"]))
    if missing:
        raise SystemExit(f"Missing from the results table: {missing}. Run notebooks/02_gate_ets.py.")
    origins = sorted(res["origin"].unique())
    cb = res[res.model == MODEL]
    print(f"\nresults table: {len(res_all)} rows for {DATASET}; {len(cb)} for {MODEL} "
          f"({cb.origin.nunique()} origins x {cb.W_est.nunique()} W_est x {cb.series_id.nunique()} "
          f"series x {cb.horizon.nunique()} horizons); NaN yhat: {int(cb.yhat.isna().sum())}; "
          f"base rows with quantiles: {int(cb.loc[cb.W_est == 'base', 'q10'].notna().sum())}")

    print("\nBase forecast diagnostics (per origin, over series x horizons):")
    print(pd.DataFrame(diag_rows).to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    coh_rows = []
    for (origin, w), g in cb.groupby(["origin", "W_est"], sort=False):
        yhat = g.pivot(index="series_id", columns="horizon", values="yhat")
        coh_rows.append({
            "origin": pd.Timestamp(origin).date(), "W_est": COLS[(MODEL, w)],
            "max_gap": np.abs(S @ yhat.loc[bottom_ids].to_numpy() - yhat.loc[ids].to_numpy()).max(),
            "n_negative": int((yhat.to_numpy() < 0).sum()),
        })
    coh = pd.DataFrame(coh_rows)
    print("\nCoherence, max |S b - y| by origin:")
    print(coh.pivot(index="origin", columns="W_est", values="max_gap")[CB_LABELS]
          .to_string(float_format=lambda x: f"{x:.3e}"))
    worst = coh[coh.W_est != "CB Base"]["max_gap"].max()
    print(f"all reconciled forecasts coherent (tolerance {COHERENCE_TOL:g}): "
          f"{'YES' if worst <= COHERENCE_TOL else 'NO'} (worst {worst:.3e})")
    print("\nNegative point forecasts by origin (count over series x horizons):")
    print(coh.pivot(index="origin", columns="W_est", values="n_negative")[CB_LABELS].to_string())

    sm = series_metrics(res)
    sm["col"] = [COLS[k] for k in zip(sm["model"], sm["W_est"])]
    level_order = list(tags)
    tags_tab = {**{k: np.asarray(v) for k, v in tags.items()}, "All series": ids}
    levels = list(tags_tab)
    lm = level_mean(sm, tags_tab, value_cols=["mase", "rmsse"], by=["origin", "col"],
                    scale_col="mase_scale", id_col="series_id")
    counts = (lm[lm.col == "CB Base"].groupby("level")
              .agg(n_series=("n_series", "first"), n_excluded=("n_excluded", "sum")).loc[levels])
    avg = lm.groupby(["level", "col"], sort=False)[["mase", "rmsse"]].mean().reset_index()

    def avg_table(value):
        return avg.pivot(index="level", columns="col", values=value).loc[levels, LABELS]

    mase_tab, rmsse_tab = avg_table("mase"), avg_table("rmsse")
    legend = "(+ = best Chronos-Bolt column, * = best of all six)"
    print(f"\nSeries excluded by the scale < 1e-8 rule, summed over origins: "
          f"{int(counts.loc['All series', 'n_excluded'])}")
    print(f"\nMean MASE by level, averaged over {len(origins)} origins   {legend}")
    print(mark_best(mase_tab, counts).to_string())
    print(f"\nMean RMSSE by level, averaged over {len(origins)} origins   {legend}")
    print(mark_best(rmsse_tab, counts).to_string())

    for name, tab in [("MASE", mase_tab), ("RMSSE", rmsse_tab)]:
        gain = pd.DataFrame({lab: (tab[lab] - tab["CB Base"]) / tab["CB Base"] for lab in CB_LABELS[1:]})
        gain["ETS MinT-shrink vs ETS Base"] = (tab["ETS MinT-shrink"] - tab["ETS Base"]) / tab["ETS Base"]
        gain["CB Base vs ETS Base"] = (tab["CB Base"] - tab["ETS Base"]) / tab["ETS Base"]
        print(f"\nRelative change in {name} (negative = better). First three columns: vs CB Base.")
        print((100 * gain).to_string(float_format=lambda x: f"{x:+.2f}%"))

    top = levels[0]
    lm_top = lm[lm.level == top].assign(origin=lambda d: d["origin"].dt.date)
    for value in ["mase", "rmsse"]:
        tab = lm_top.pivot(index="origin", columns="col", values=value)[LABELS]
        tab.loc["mean"] = tab.mean()
        print(f"\n{top} {value.upper()} by origin   {legend}")
        print(mark_best(tab).to_string())

    bias = sm[sm.level == top].assign(origin=lambda d: d["origin"].dt.date)
    bias_tab = bias.pivot(index="origin", columns="col", values="bias")[LABELS]
    mean_y = (res[(res.level == top) & (res.model == MODEL) & (res.W_est == "base")]
              .groupby("origin")["y"].mean().rename(lambda o: pd.Timestamp(o).date()))
    out = pd.DataFrame({"mean_y": mean_y})
    for lab in LABELS:
        out[f"bias {lab}"] = bias_tab[lab]
    print(f"\n{top} bias by origin: mean over the 12 horizons of (yhat - y)")
    print(out.to_string(float_format=lambda x: f"{x:.1f}"))
    pct = pd.DataFrame({lab: 100 * bias_tab[lab] / mean_y for lab in LABELS})
    print(f"\n{top} bias as % of the mean actual")
    print(pct.to_string(float_format=lambda x: f"{x:+.2f}%"))

    # bias at the bottom level, to see what the reconciliation is pulling toward
    bottom_level = level_order[-1]
    bsum = (res[(res.level == bottom_level) & (res.W_est == "base")]
            .assign(e=lambda d: d["yhat"] - d["y"])
            .groupby(["model", "origin"])["e"].sum() / h)
    bsum = bsum.unstack("model").rename(index=lambda o: pd.Timestamp(o).date())
    print(f"\nSum over the {nb} bottom series of the base bias (= Country bias of BottomUp):")
    print(bsum.to_string(float_format=lambda x: f"{x:.1f}"))

    # ---- descriptive diagnostics for WLS-pv (exploratory, not a registered test) ----
    cbb = res[(res.model == MODEL) & (res.W_est == "base")].copy()
    cbb["v"] = predictive_variance(cbb[QUANTILE_COLS].to_numpy())
    cbb["se"] = (cbb["yhat"] - cbb["y"]) ** 2
    cbb["qmean"] = cbb[QUANTILE_COLS].mean(axis=1)
    g = cbb.groupby("level", sort=False)
    diag = pd.DataFrame({
        "rms_pred_sd": np.sqrt(g["v"].mean()),
        "rmse_base": np.sqrt(g["se"].mean()),
    }).loc[level_order]
    diag["ratio"] = diag["rms_pred_sd"] / diag["rmse_base"]
    diag["share_of_total_weight_1/v"] = (1 / cbb["v"]).groupby(cbb["level"], sort=False).sum() \
        / (1 / cbb["v"]).sum()
    print(f"\nExploratory: {MODEL} predictive sd (from q10, q90) against the realised RMSE of the "
          "base forecast, pooled over origins and horizons")
    print(diag.to_string(float_format=lambda x: f"{x:.3f}"))

    ets_base = res[(res.model == "AutoETS") & (res.W_est == "base")]
    add = pd.DataFrame({
        "actual": cbb[cbb.level == top].groupby("origin")["y"].mean(),
        "CB Country median": cbb[cbb.level == top].groupby("origin")["yhat"].mean(),
        "CB sum of bottom medians": cbb[cbb.level == bottom_level].groupby("origin")["yhat"].sum() / h,
        "CB sum of bottom quantile means": cbb[cbb.level == bottom_level].groupby("origin")["qmean"].sum() / h,
        "ETS Country": ets_base[ets_base.level == top].groupby("origin")["yhat"].mean(),
        "ETS sum of bottom": ets_base[ets_base.level == bottom_level].groupby("origin")["yhat"].sum() / h,
    }).rename(index=lambda o: pd.Timestamp(o).date())
    print("\nExploratory: do the bottom forecasts add up to the Country forecast? "
          "Monthly averages over the 12 horizons")
    print(add.to_string(float_format=lambda x: f"{x:.0f}"))
    print((100 * add.drop(columns="actual").sub(add["actual"], axis=0).div(add["actual"], axis=0))
          .to_string(float_format=lambda x: f"{x:+.1f}%"))
    skew = cbb[cbb.level == bottom_level]
    print(f"bottom level: share of forecasts with median < mean of the 9 quantiles: "
          f"{(skew['yhat'] < skew['qmean']).mean():.3f}")

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
          f"{runs[['ols_s', 'wls_struct_s', 'wls_pv_s']].sum().sum():.2f}s; metrics {t_metrics:.2f}s; "
          f"this run wall clock {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
