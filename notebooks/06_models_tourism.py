"""Weeks 4-5: every model on TourismLarge, all six W-estimators, all point arms.

Reads only cached forecasts (scripts/run_plan.sh produces them) and skips any
model whose five origins are not yet cached. Reconciles with the numpy
reconciler, the predictive-variance floor applied everywhere (Deviations log),
appends to results/results.parquet and reports, per model, MASE and RMSSE for
"All series" and Country per W and per arm, plus the Country bias. The full
level tables go to results/level_tables_TourismLarge.parquet and are not printed.

Arms: median (registered), mean (exploratory for Chronos-Bolt-small, pre-specified
for the others), mean9 for Chronos-2 (sensitivity). Chronos-T5 is base only:
OLS, WLS-struct and WLS-pv, with the sample variance as its predictive variance.

Do not import hierarchicalforecast here (see src/runners.py).

Run from the repo root: python notebooks/06_models_tourism.py [model ...]
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
    W_hybrid, W_ols, W_pv, W_shrink_bt, W_struct, W_var_bt, predictive_variance, pv_floor, reconcile,
    reconcile_by_horizon,
)
from src.residuals import backtest_paths, backtest_residuals, residual_array  # noqa: E402
from src.results import append_results, build_results  # noqa: E402
from src.runners import MODELS  # noqa: E402

DATASET = "TourismLarge"
K, N_INNER = 5, 24
TABLE = ROOT / "results" / "results.parquet"
LEVEL_TABLES = ROOT / "results" / f"level_tables_{DATASET}.parquet"
ORDER = ["chronos_bolt_tiny", "chronos_bolt_mini", "chronos_bolt_small", "moirai_2_small", "chronos_2",
         "chronos_t5_small", "tirex"]
BASE_ONLY = {"chronos_t5_small"}
ARMS = {m: ["median", "mean"] for m in ORDER}
ARMS["chronos_2"] = ["median", "mean", "mean9"]
W_LABEL = {"base": "Base", "ols": "OLS", "wls_struct": "WLS-struct", "wls_pv": "WLS-pv",
           "wls_var_bt": "WLS-var_bt", "mint_shrink_bt": "MinT-shrink_bt", "hybrid": "Hybrid"}
W_FULL = list(W_LABEL)
W_BASE_ONLY = ["base", "ols", "wls_struct", "wls_pv"]


def cached(model, splits):
    for s in splits:
        if not cache_paths(DATASET, s["origin"], model)[0].exists():
            return False
        if model not in BASE_ONLY and not backtest_paths(DATASET, s["origin"], model)["inner"].exists():
            return False
    return True


def reconcile_model(model, splits, S, ids, wide, m, tags, freq, h):
    """Reconcile one model at every origin and arm. Returns results rows and diagnostics."""
    n = S.shape[0]
    w_ests = W_BASE_ONLY if model in BASE_ONLY else W_FULL
    rows, diag = [], []
    for split in splits:
        origin, train, test = split["origin"], split["train"], split["test"]
        base = forecast(model, train, h, freq=freq, dataset=DATASET)

        def to_wide(values):
            return (base.assign(_v=values).pivot(index="unique_id", columns="ds", values="_v")
                    .loc[ids].to_numpy())

        dates = np.sort(base["ds"].unique())
        y_train = wide.loc[:, :origin].to_numpy()
        rmsse_scale = scale_rmsse(y_train, m)
        scales = pd.DataFrame({"unique_id": ids, "mase_scale": scale_mase(y_train, m),
                               "rmsse_scale": rmsse_scale})
        floor = pv_floor(rmsse_scale)
        if MODELS[model]["output"] == "samples":
            v = to_wide(base["sample_var"].to_numpy())                       # (n, h)
            q = None
        else:
            q = np.stack([to_wide(base[c].to_numpy()) for c in QUANTILE_COLS], axis=-1)
            v = predictive_variance(q)
        floored = int((v < floor[:, None]).sum())
        Ws_pv = W_pv(variance=v, floor=floor)
        quant = base[["unique_id", "ds", *QUANTILE_COLS]].assign(W_est="base")

        for point in ARMS[model]:
            y_hat = to_wide(point_forecast(base, point))
            rec = {"base": y_hat,
                   "ols": reconcile(y_hat, S, W_ols(n)),
                   "wls_struct": reconcile(y_hat, S, W_struct(S)),
                   "wls_pv": reconcile_by_horizon(y_hat, S, Ws_pv)}
            lams = None
            if model not in BASE_ONLY:
                resid_df = backtest_residuals(model, train, h, N_INNER, point=point, freq=freq,
                                              dataset=DATASET)
                if resid_df["ds"].max() > origin:
                    raise RuntimeError("A backtest residual target lies after the outer origin.")
                resid = residual_array(resid_df, ids)
                rec["wls_var_bt"] = reconcile_by_horizon(y_hat, S, W_var_bt(resid))
                Ws, lams = W_shrink_bt(resid, return_lambda=True)
                rec["mint_shrink_bt"] = reconcile_by_horizon(y_hat, S, Ws)
                rec["hybrid"] = reconcile_by_horizon(y_hat, S, W_hybrid(resid=resid, variance=v, floor=floor))
            fc = pd.DataFrame({"unique_id": np.repeat(ids, h), "ds": np.tile(dates, n)})
            for w in w_ests:
                fc[w] = rec[w].reshape(-1)
            rows.append(build_results(DATASET, origin, model, fc, test, tags, scales,
                                      {w: w for w in w_ests}, point=point, quantiles=quant))
            diag.append({"model": model, "origin": origin.date(), "point": point,
                         "floored_cells": floored, "cells": int(v.size),
                         "lambda_min": min(lams) if lams else np.nan,
                         "lambda_median": float(np.median(lams)) if lams else np.nan,
                         "lambda_max": max(lams) if lams else np.nan,
                         "negative_base": int((y_hat < 0).sum())})
    return rows, pd.DataFrame(diag)


def main():
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    t_start = time.perf_counter()
    Y_df, S_df, tags, freq, h = load_hierarchy(DATASET)
    m = DATASETS[DATASET]["m"]
    ids = S_df["unique_id"].to_numpy()
    bottom_ids = [c for c in S_df.columns if c != "unique_id"]
    S = S_df[bottom_ids].to_numpy(dtype=np.float64)
    wide = Y_df.pivot(index="unique_id", columns="ds", values="y").loc[ids]
    splits = rolling_origins(Y_df, h, K)
    top = list(tags)[0]
    tags_tab = {**{k2: np.asarray(v2) for k2, v2 in tags.items()}, "All series": ids}

    wanted = sys.argv[1:] or ORDER
    ready = [mo for mo in wanted if cached(mo, splits)]
    missing = [mo for mo in wanted if mo not in ready]
    print(f"{DATASET}: models with all {K} origins cached: {ready}; not yet: {missing}")

    # Week 3 numbers of Chronos-Bolt-small, before the floor, for the comparison
    before = None
    if TABLE.exists() and "chronos_bolt_small" in ready:
        old = pd.read_parquet(TABLE)
        old = old[(old.dataset == DATASET) & (old.model == "chronos_bolt_small")]
        if len(old):
            sm_old = series_metrics(old)
            before = level_mean(sm_old, tags_tab, value_cols=["mase", "rmsse"], by=["origin", "point", "W_est"],
                                scale_col="mase_scale", id_col="series_id")
            before = before.groupby(["point", "W_est", "level"], sort=False)[["mase", "rmsse"]].mean()

    diags = []
    for model in ready:
        t0 = time.perf_counter()
        rows, diag = reconcile_model(model, splits, S, ids, wide, m, tags, freq, h)
        append_results(TABLE, pd.concat(rows, ignore_index=True))
        diags.append(diag)
        print(f"  {model}: reconciled and appended in {time.perf_counter() - t0:.1f}s", flush=True)
    diag = pd.concat(diags, ignore_index=True) if diags else pd.DataFrame()

    # ---- everything below reads only the results table ----
    res = pd.read_parquet(TABLE)
    res = res[(res.dataset == DATASET) & res.model.isin(ready + ["AutoETS"])]
    sm = series_metrics(res)
    lm = level_mean(sm, tags_tab, value_cols=["mase", "rmsse"], by=["origin", "model", "point", "W_est"],
                    scale_col="mase_scale", id_col="series_id")
    level_tables = (lm.groupby(["model", "point", "W_est", "level"], sort=False)
                    .agg(mase=("mase", "mean"), rmsse=("rmsse", "mean"), n_series=("n_series", "first"),
                         n_excluded=("n_excluded", "sum"), n_origins=("origin", "nunique")).reset_index())
    level_tables.to_parquet(LEVEL_TABLES, index=False)
    print(f"\nfull level tables: {LEVEL_TABLES.relative_to(ROOT)} ({len(level_tables)} rows, not printed)")
    print(f"series excluded by the scale rule: {int(level_tables.n_excluded.sum())}")

    coh = []
    for (model, origin, point, w), g in res[res.model != "AutoETS"].groupby(
            ["model", "origin", "point", "W_est"], sort=False):
        yh = g.pivot(index="series_id", columns="horizon", values="yhat")
        coh.append({"model": model, "W_est": w,
                    "gap": np.abs(S @ yh.loc[bottom_ids].to_numpy() - yh.loc[ids].to_numpy()).max()})
    coh = pd.DataFrame(coh)
    worst = coh[coh.W_est != "base"].gap.max()
    print(f"coherence of every reconciled forecast, every model, origin and arm: worst gap {worst:.2e} "
          f"({'YES' if worst <= 1e-8 else 'NO'})")

    ets = level_tables[level_tables.model == "AutoETS"].set_index(["W_est", "level"])
    print("\nReference, AutoETS (point = mean), 5-origin mean:")
    print(pd.DataFrame({
        "All series MASE": [ets.loc[("base", "All series"), "mase"], ets.loc[("mint_shrink", "All series"), "mase"]],
        "Country MASE": [ets.loc[("base", top), "mase"], ets.loc[("mint_shrink", top), "mase"]],
        "All series RMSSE": [ets.loc[("base", "All series"), "rmsse"], ets.loc[("mint_shrink", "All series"), "rmsse"]],
        "Country RMSSE": [ets.loc[("base", top), "rmsse"], ets.loc[("mint_shrink", top), "rmsse"]],
    }, index=["ETS Base", "ETS MinT-shrink"]).to_string(float_format=lambda x: f"{x:.4f}"))

    bias = sm[sm.level == top].groupby(["model", "point", "W_est"], sort=False)["bias"].mean()
    mean_y = res[(res.level == top) & (res.W_est == "base")].groupby("origin")["y"].mean().mean()
    for model in ready:
        lt = level_tables[level_tables.model == model].set_index(["point", "W_est", "level"])
        w_ests = W_BASE_ONLY if model in BASE_ONLY else W_FULL
        print("\n" + "=" * 120 + f"\n{model}" + ("   (base only)" if model in BASE_ONLY else "")
              + f"   arms: {', '.join(ARMS[model])}" + ("   [mean arm exploratory]" if model == "chronos_bolt_small" else "")
              + "\n" + "=" * 120)
        cols = {}
        for point in ARMS[model]:
            for value, lvl, name in [("mase", "All series", "All MASE"), ("mase", top, "Country MASE"),
                                     ("rmsse", "All series", "All RMSSE"), ("rmsse", top, "Country RMSSE")]:
                cols[f"{point}: {name}"] = [lt.loc[(point, w, lvl), value] for w in w_ests]
        tab = pd.DataFrame(cols, index=[W_LABEL[w] for w in w_ests])
        out = tab.map(lambda x: f"{x:.4f}")
        for c in tab.columns:
            out.loc[tab[c].idxmin(), c] += "*"
        print(out.to_string())
        b = pd.DataFrame({point: [bias.loc[(model, point, w)] for w in w_ests] for point in ARMS[model]},
                         index=[W_LABEL[w] for w in w_ests])
        print(f"\nCountry bias, mean over origins and horizons of (yhat - y); mean actual {mean_y:.0f}:")
        print(b.to_string(float_format=lambda x: f"{x:.0f}"))
        d = diag[diag.model == model]
        d1 = d[d.point == ARMS[model][0]]
        print(f"floored predictive variances: {int(d1.floored_cells.sum())} of {int(d1.cells.sum())} "
              f"series x horizon cells over the {K} origins (same for every arm); "
              + (f"lambda over origins and horizons: min {d.lambda_min.min():.3f}, median "
                 f"{d.lambda_median.median():.3f}, max {d.lambda_max.max():.3f}" if model not in BASE_ONLY else
                 "no backtests (base only)"))
        print("floored cells by origin: " + ", ".join(
            f"{r.origin}: {r.floored_cells}" for r in d[d.point == ARMS[model][0]].itertuples()))

    if before is not None:
        print("\n" + "=" * 120 + "\nChronos-Bolt-small: Week 3 (no floor) against now (floor), 5-origin means\n" + "=" * 120)
        now = level_tables[level_tables.model == "chronos_bolt_small"].set_index(["point", "W_est", "level"])
        cmp = before.join(now[["mase", "rmsse"]], rsuffix="_now", how="inner")
        cmp["d_mase"] = cmp["mase_now"] - cmp["mase"]
        cmp["d_rmsse"] = cmp["rmsse_now"] - cmp["rmsse"]
        changed = cmp[(cmp.d_mase.abs() > 1e-12) | (cmp.d_rmsse.abs() > 1e-12)]
        print(f"rows compared: {len(cmp)}; rows that changed: {len(changed)}; "
              f"largest |change| MASE {cmp.d_mase.abs().max():.2e}, RMSSE {cmp.d_rmsse.abs().max():.2e}")
        if len(changed):
            print(changed.reset_index().groupby(["point", "W_est"]).agg(
                levels_changed=("level", "size"), max_d_mase=("d_mase", lambda x: x.abs().max()),
                max_d_rmsse=("d_rmsse", lambda x: x.abs().max())).to_string(float_format=lambda x: f"{x:.2e}"))

    print(f"\nwall clock {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
