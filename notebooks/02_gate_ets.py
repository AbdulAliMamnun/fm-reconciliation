"""Correctness gate, part 1 (PREREG §8.1 as revised in the Deviations log).

AutoETS + hierarchicalforecast on TourismLarge over the 5 pre-registered
rolling origins (h=12, origins 12 months apart, ending at the last
observation). Base forecasts, fitted values and reconciled forecasts are
cached per origin in results/ and reused if present. Every forecast goes into
the long results table (results/results.parquet); all metrics and checks below
are computed from that table.

Run from the repo root: python notebooks/02_gate_ets.py [--per-origin]
  --per-origin  also print the full MASE and RMSSE tables for each origin
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_hierarchy  # noqa: E402
from src.metrics import level_mean, scale_mase, scale_rmsse, series_metrics  # noqa: E402
from src.results import append_results, build_results  # noqa: E402

DATASET = "TourismLarge"
MODEL = "AutoETS"
M = 12
K = 5
COHERENCE_TOL = 1e-8
RESULTS = ROOT / "results"
TABLE = RESULTS / "results.parquet"

# Column produced by hierarchicalforecast -> W_est in the results table.
W_EST = {
    "AutoETS": "base",
    "AutoETS/BottomUp": "bottomup",
    "AutoETS/MinTrace_method-ols": "ols",
    "AutoETS/MinTrace_method-wls_struct": "wls_struct",
    "AutoETS/MinTrace_method-wls_var": "wls_var",
    "AutoETS/MinTrace_method-mint_shrink": "mint_shrink",
}
LABELS = {
    "base": "Base", "bottomup": "BottomUp", "ols": "OLS",
    "wls_struct": "WLS-struct", "wls_var": "WLS-var", "mint_shrink": "MinT-shrink",
}


def forecast_origin(train, S_df, tags, freq, h, key):
    """Base + reconciled forecasts for one origin, from cache if present."""
    base_path = RESULTS / f"base_{key}.parquet"
    fitted_path = RESULTS / f"fitted_{key}.parquet"
    rec_path = RESULTS / f"rec_{key}_hf.parquet"
    timing_path = RESULTS / f"timing_{key}.parquet"
    timings = (
        pd.read_parquet(timing_path).set_index("stage")["seconds"].to_dict()
        if timing_path.exists() else {}
    )
    status = []

    if base_path.exists() and fitted_path.exists():
        Y_hat_df = pd.read_parquet(base_path)
        Y_fitted_df = pd.read_parquet(fitted_path)
        status.append("base: cache")
    else:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS

        t0 = time.perf_counter()
        sf = StatsForecast(models=[AutoETS(season_length=M)], freq=freq, n_jobs=-1)
        Y_hat_df = sf.forecast(df=train, h=h, fitted=True)
        Y_fitted_df = sf.forecast_fitted_values()
        timings["base_forecast"] = time.perf_counter() - t0
        Y_hat_df.to_parquet(base_path, index=False)
        Y_fitted_df.to_parquet(fitted_path, index=False)
        status.append("base: fitted")

    if rec_path.exists():
        Y_rec_df = pd.read_parquet(rec_path)
        status.append("reconciled: cache")
    else:
        from hierarchicalforecast.core import HierarchicalReconciliation
        from hierarchicalforecast.methods import BottomUp, MinTrace

        reconcilers = [
            BottomUp(),
            MinTrace(method="ols", nonnegative=False),
            MinTrace(method="wls_struct", nonnegative=False),
            MinTrace(method="wls_var", nonnegative=False),
            MinTrace(method="mint_shrink", nonnegative=False),
        ]
        t0 = time.perf_counter()
        hrec = HierarchicalReconciliation(reconcilers=reconcilers)
        Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_fitted_df, S_df=S_df, tags=tags)
        timings["reconciliation"] = time.perf_counter() - t0
        Y_rec_df.to_parquet(rec_path, index=False)
        status.append("reconciled: computed")

    pd.DataFrame({"stage": list(timings), "seconds": list(timings.values())}).to_parquet(
        timing_path, index=False
    )
    n_nan_fitted = int(Y_fitted_df[MODEL].isna().sum())
    return Y_rec_df, timings, ", ".join(status), n_nan_fitted


def mark_best(tab, counts=None):
    best = tab.idxmin(axis=1)
    out = tab.map(lambda x: f"{x:.4f}")
    for idx in tab.index:
        out.loc[idx, best[idx]] += "*"
    if counts is not None:
        out.insert(0, "excl", counts["n_excluded"])
        out.insert(0, "n", counts["n_series"])
    return out


def yes(flag):
    return "YES" if flag else "NO"


def main():
    per_origin = "--per-origin" in sys.argv
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    RESULTS.mkdir(exist_ok=True)
    t_start = time.perf_counter()

    Y_df, S_df, tags, freq, h = load_hierarchy(DATASET)
    ids = S_df["unique_id"].to_numpy()
    bottom_ids = [c for c in S_df.columns if c != "unique_id"]
    S = S_df[bottom_ids].to_numpy()
    dates = np.sort(Y_df["ds"].unique())
    T = len(dates)
    wide = Y_df.pivot(index="unique_id", columns="ds", values="y").loc[ids]

    # ---- forecasts, per origin, into the long results table ----
    print(f"{DATASET}: {len(ids)} series, {K} origins, h={h}, model {MODEL}")
    timing_rows, new_rows = [], []
    for k in range(K, 0, -1):  # earliest origin first
        n_train = T - h * k
        origin = pd.Timestamp(dates[n_train - 1])
        test_dates = dates[n_train : n_train + h]
        train = Y_df[Y_df["ds"] <= origin]
        actuals = Y_df[Y_df["ds"].isin(test_dates)]
        key = f"{DATASET}_{origin.date()}_{MODEL}"

        Y_rec_df, timings, status, n_nan_fitted = forecast_origin(train, S_df, tags, freq, h, key)
        print(f"  origin {origin.date()}: train {n_train} months, test "
              f"{pd.Timestamp(test_dates[0]).date()} to {pd.Timestamp(test_dates[-1]).date()} "
              f"[{status}; NaN fitted values: {n_nan_fitted}]")
        timing_rows.append({"origin": origin.date(), **timings})

        y_train = wide.iloc[:, :n_train].to_numpy()
        scales = pd.DataFrame({
            "unique_id": ids,
            "mase_scale": scale_mase(y_train, M),
            "rmsse_scale": scale_rmsse(y_train, M),
        })
        new_rows.append(build_results(DATASET, origin, MODEL, Y_rec_df, actuals, tags, scales, W_EST))

    append_results(TABLE, pd.concat(new_rows, ignore_index=True))

    # ---- everything below reads only the results table ----
    t0 = time.perf_counter()
    res = pd.read_parquet(TABLE)
    res = res[(res["dataset"] == DATASET) & (res["model"] == MODEL)]
    origins = sorted(res["origin"].unique())
    w_ests = list(LABELS)
    labels = list(LABELS.values())
    print(f"\nresults table: {TABLE.relative_to(ROOT)}, {len(res)} rows for {DATASET}/{MODEL} "
          f"({len(origins)} origins x {res.W_est.nunique()} W_est x {res.series_id.nunique()} series "
          f"x {res.horizon.nunique()} horizons), NaN yhat: {int(res.yhat.isna().sum())}")

    # (i) coherence: max over horizons and series of |S b - y|
    coh_rows = []
    for (origin, w), g in res.groupby(["origin", "W_est"], sort=False):
        yhat = g.pivot(index="series_id", columns="horizon", values="yhat")
        gap = np.abs(S @ yhat.loc[bottom_ids].to_numpy() - yhat.loc[ids].to_numpy()).max()
        coh_rows.append({"origin": pd.Timestamp(origin).date(), "W_est": LABELS[w], "max_gap": gap,
                         "n_negative": int((yhat.to_numpy() < 0).sum())})
    coh = pd.DataFrame(coh_rows)
    coh_tab = coh.pivot(index="origin", columns="W_est", values="max_gap")[labels]
    neg_tab = coh.pivot(index="origin", columns="W_est", values="n_negative")[labels]

    # per-series metrics -> mean within level -> mean over origins
    sm = series_metrics(res)
    level_order = list(dict.fromkeys(res["level"]))
    tags_tab = {lvl: sm.loc[sm.level == lvl, "series_id"].unique() for lvl in level_order}
    tags_tab["All series"] = sm["series_id"].unique()
    levels = list(tags_tab)
    lm = level_mean(sm, tags_tab, value_cols=["mase", "rmsse"], by=["origin", "W_est"],
                    scale_col="mase_scale", id_col="series_id")
    lm["method"] = lm["W_est"].map(LABELS)

    counts = (lm[lm.W_est == "base"].groupby("level")[["n_series", "n_excluded"]]
              .agg(n_series=("n_series", "first"), n_excluded=("n_excluded", "sum")).loc[levels])
    avg = lm.groupby(["level", "method"], sort=False)[["mase", "rmsse"]].mean().reset_index()

    def avg_table(value):
        return avg.pivot(index="level", columns="method", values=value).loc[levels, labels]

    mase_tab, rmsse_tab = avg_table("mase"), avg_table("rmsse")

    print(f"\nSeries excluded by the scale < 1e-8 rule, summed over origins: "
          f"{int(counts.loc['All series', 'n_excluded'])}")
    print(f"\nMean MASE by level, averaged over {len(origins)} origins   (* = best in row)")
    print(mark_best(mase_tab, counts).to_string())
    print(f"\nMean RMSSE by level, averaged over {len(origins)} origins   (* = best in row)")
    print(mark_best(rmsse_tab, counts).to_string())

    # Country row per origin
    top = levels[0]
    lm_top = lm[lm.level == top].assign(origin=lambda d: d["origin"].dt.date)
    for value in ["mase", "rmsse"]:
        tab = lm_top.pivot(index="origin", columns="method", values=value)[labels]
        tab.loc["mean"] = tab.mean()
        print(f"\n{top} {value.upper()} by origin   (* = best in row)")
        print(mark_best(tab).to_string())

    # ---- gate verdict (revised §8.1) ----
    print("\n" + "=" * 100)
    print("GATE VERDICT (PREREG §8.1 as revised in the Deviations log), 5-origin average")
    print("=" * 100)

    rec_labels = [lab for lab in labels if lab != "Base"]
    worst = coh_tab[rec_labels].max().max()
    pass_i = bool(worst <= COHERENCE_TOL)
    print(f"\n(i) All reconciled forecasts coherent: {yes(pass_i)}")
    print(f"    max |S b - y| over all reconciled methods, origins, horizons: {worst:.3e} "
          f"(tolerance {COHERENCE_TOL:g})")
    print("    max |S b - y| by origin and method:")
    print(coh_tab.to_string(float_format=lambda x: f"{x:.3e}"))

    cmp = pd.DataFrame({
        "Base": mase_tab["Base"], "MinT-shrink": mase_tab["MinT-shrink"],
        "diff": mase_tab["MinT-shrink"] - mase_tab["Base"],
        "rel_gain": (mase_tab["MinT-shrink"] - mase_tab["Base"]) / mase_tab["Base"],
        "MinT<=Base": mase_tab["MinT-shrink"] <= mase_tab["Base"],
    })
    n_levels_ok = int(cmp.loc[level_order, "MinT<=Base"].sum())
    all_ok = bool(cmp.loc["All series", "MinT<=Base"])
    pass_ii = all_ok and n_levels_ok >= 5
    print(f"\n(ii) MinT-shrink mean MASE <= Base on 'All series' and at >= 5 of 8 levels: {yes(pass_ii)}")
    print(f"    All series: {yes(all_ok)} ({cmp.loc['All series', 'MinT-shrink']:.4f} vs "
          f"{cmp.loc['All series', 'Base']:.4f}); levels: {n_levels_ok} of {len(level_order)}")
    print(cmp.to_string(float_format=lambda x: f"{x:+.4f}"))

    bu, mint = mase_tab.loc[top, "BottomUp"], mase_tab.loc[top, "MinT-shrink"]
    pass_iii = bool(bu > mint)
    print(f"\n(iii) BottomUp worse than MinT-shrink at {top}: {yes(pass_iii)}")
    print(f"    MASE: BottomUp {bu:.4f} vs MinT-shrink {mint:.4f}")
    print(f"    RMSSE (secondary): BottomUp {rmsse_tab.loc[top, 'BottomUp']:.4f} vs "
          f"MinT-shrink {rmsse_tab.loc[top, 'MinT-shrink']:.4f}")
    lm_piv = lm_top.pivot(index="origin", columns="method", values="mase")
    n_origins_bu_worse = int((lm_piv["BottomUp"] > lm_piv["MinT-shrink"]).sum())
    print(f"    origins where BottomUp is worse (MASE): {n_origins_bu_worse} of {len(origins)}")

    print(f"\nPart 1 of the gate: (i) {yes(pass_i)}, (ii) {yes(pass_ii)}, (iii) {yes(pass_iii)}. "
          f"§8.2 (numpy reconciler) is not tested here.")

    # ---- original chain, descriptive only ----
    for name, tab in [("MASE", mase_tab), ("RMSSE", rmsse_tab)]:
        chain = pd.DataFrame({
            "MinT<=WLS-var": tab["MinT-shrink"] <= tab["WLS-var"],
            "MinT<=WLS-struct": tab["MinT-shrink"] <= tab["WLS-struct"],
            "WLS-var<=OLS": tab["WLS-var"] <= tab["OLS"],
            "WLS-struct<=OLS": tab["WLS-struct"] <= tab["OLS"],
            "OLS<=Base": tab["OLS"] <= tab["Base"],
        })
        chain["chain_wls_var"] = chain[["MinT<=WLS-var", "WLS-var<=OLS", "OLS<=Base"]].all(axis=1)
        chain["chain_wls_struct"] = chain[["MinT<=WLS-struct", "WLS-struct<=OLS", "OLS<=Base"]].all(axis=1)
        chain["gain_MinT_vs_Base"] = (tab["MinT-shrink"] - tab["Base"]) / tab["Base"]
        print(f"\nDescriptive: original chain MinT-shrink <= WLS <= OLS <= Base, {name}, 5-origin average")
        print(chain.to_string(float_format=lambda x: f"{x:+.4f}"))

    # ---- Country bias ----
    bias = sm[sm.level == top].assign(origin=lambda d: d["origin"].dt.date,
                                      method=lambda d: d["W_est"].map(LABELS))
    mean_y = (res[res.level == top].groupby("origin")["y"].mean()
              .rename(lambda o: pd.Timestamp(o).date()))
    bias_tab = bias.pivot(index="origin", columns="method", values="bias")[labels]
    out = pd.DataFrame({"mean_y": mean_y})
    for lab in ["Base", "BottomUp", "MinT-shrink"]:
        out[f"bias_{lab}"] = bias_tab[lab]
        out[f"pct_{lab}"] = 100 * bias_tab[lab] / mean_y
    out["MASE_Base"] = lm_piv["Base"]
    out["MASE_BottomUp"] = lm_piv["BottomUp"]
    out["MASE_MinT-shrink"] = lm_piv["MinT-shrink"]
    print(f"\n{top} bias by origin: mean over the 12 horizons of (yhat - y); pct = bias / mean actual")
    print(out.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nNegative forecasts by origin and method (count over series x horizons):")
    print(neg_tab.to_string())

    if per_origin:
        for origin in origins:
            sub = lm[lm.origin == origin]
            for value in ["mase", "rmsse"]:
                tab = sub.pivot(index="level", columns="method", values=value).loc[levels, labels]
                print(f"\nOrigin {pd.Timestamp(origin).date()}: mean {value.upper()} by level   (* = best in row)")
                print(mark_best(tab).to_string())

    # ---- runtime ----
    t_metrics = time.perf_counter() - t0
    tt = pd.DataFrame(timing_rows).set_index("origin")
    print("\nRuntime in seconds (forecast and reconciliation times were measured when each cache was written):")
    print(tt.to_string(float_format=lambda x: f"{x:.1f}"))
    print(f"  total base_forecast {tt['base_forecast'].sum():.1f}s, total reconciliation "
          f"{tt['reconciliation'].sum():.1f}s, metrics and checks {t_metrics:.1f}s, "
          f"this run wall clock {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
