"""Correctness gate, part 1 (PREREG §8.1): AutoETS + hierarchicalforecast on TourismLarge.

Single origin, last h=12 months held out. Base forecasts, fitted values and
reconciled forecasts are cached to results/ as parquet and reused if present.

Run from the repo root: python notebooks/02_gate_ets.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_hierarchy  # noqa: E402
from src.metrics import level_mean, mase, rmsse, scale_mase, scale_rmsse  # noqa: E402

DATASET = "TourismLarge"
MODEL = "AutoETS"
M = 12
RESULTS = ROOT / "results"

# Column produced by hierarchicalforecast -> label in the tables.
METHODS = {
    "AutoETS": "Base",
    "AutoETS/BottomUp": "BottomUp",
    "AutoETS/MinTrace_method-ols": "OLS",
    "AutoETS/MinTrace_method-wls_struct": "WLS-struct",
    "AutoETS/MinTrace_method-wls_var": "WLS-var",
    "AutoETS/MinTrace_method-mint_shrink": "MinT-shrink",
}


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    RESULTS.mkdir(exist_ok=True)

    Y_df, S_df, tags, freq, h = load_hierarchy(DATASET)
    dates = np.sort(Y_df["ds"].unique())
    origin = pd.Timestamp(dates[-h - 1])  # last training timestamp
    train = Y_df[Y_df["ds"] <= origin]
    test = Y_df[Y_df["ds"] > origin]
    key = f"{DATASET}_{origin.date()}_{MODEL}"
    print(f"{DATASET}: {S_df.shape[0]} series, train {train.ds.min().date()} to {origin.date()} "
          f"({train.ds.nunique()} months), test {test.ds.min().date()} to {test.ds.max().date()} (h={h})")

    base_path = RESULTS / f"base_{key}.parquet"
    fitted_path = RESULTS / f"fitted_{key}.parquet"
    rec_path = RESULTS / f"rec_{key}_hf.parquet"
    timing_path = RESULTS / f"timing_{key}.parquet"
    timings = (
        pd.read_parquet(timing_path).set_index("stage")["seconds"].to_dict()
        if timing_path.exists() else {}
    )

    # ---- base forecasts ----
    if base_path.exists() and fitted_path.exists():
        print(f"base forecasts: loaded from cache ({base_path.name})")
        Y_hat_df = pd.read_parquet(base_path)
        Y_fitted_df = pd.read_parquet(fitted_path)
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
        print(f"base forecasts: fitted in {timings['base_forecast']:.1f}s")

    # ---- reconciliation ----
    if rec_path.exists():
        print(f"reconciled forecasts: loaded from cache ({rec_path.name})")
        Y_rec_df = pd.read_parquet(rec_path)
    else:
        from hierarchicalforecast.core import HierarchicalReconciliation
        from hierarchicalforecast.methods import BottomUp, MinTrace

        reconcilers = [
            BottomUp(),
            MinTrace(method="ols"),
            MinTrace(method="wls_struct"),
            MinTrace(method="wls_var"),
            MinTrace(method="mint_shrink"),
        ]
        t0 = time.perf_counter()
        hrec = HierarchicalReconciliation(reconcilers=reconcilers)
        Y_rec_df = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_fitted_df, S_df=S_df, tags=tags)
        timings["reconciliation"] = time.perf_counter() - t0
        Y_rec_df.to_parquet(rec_path, index=False)
        print(f"reconciliation: done in {timings['reconciliation']:.1f}s")

    pd.DataFrame({"stage": list(timings), "seconds": list(timings.values())}).to_parquet(
        timing_path, index=False
    )

    missing = [c for c in METHODS if c not in Y_rec_df.columns]
    if missing:
        raise RuntimeError(f"Missing columns {missing}; got {list(Y_rec_df.columns)}")

    # ---- sanity checks on the forecasts ----
    ids = S_df["unique_id"].to_numpy()
    bottom_ids = [c for c in S_df.columns if c != "unique_id"]
    S = S_df[bottom_ids].to_numpy()
    print()
    print("Checks:")
    print(f"  NaN in forecasts: {int(Y_rec_df[list(METHODS)].isna().sum().sum())}")
    print(f"  NaN in fitted values: {int(Y_fitted_df[MODEL].isna().sum())} "
          f"of {len(Y_fitted_df)} ({Y_fitted_df.ds.nunique()} months per series)")
    for col, label in METHODS.items():
        wide = Y_rec_df.pivot(index="unique_id", columns="ds", values=col)
        incoherence = np.abs(S @ wide.loc[bottom_ids].to_numpy() - wide.loc[ids].to_numpy()).max()
        n_neg = int((wide.to_numpy() < 0).sum())
        print(f"  {label:<12} max |S b - y| = {incoherence:.3e}   negative forecasts: {n_neg}")

    # ---- metrics: per series, then mean within level ----
    t0 = time.perf_counter()
    train_wide = train.pivot(index="unique_id", columns="ds", values="y").loc[ids].to_numpy()
    test_wide = test.pivot(index="unique_id", columns="ds", values="y").loc[ids]
    scale = scale_mase(train_wide, M)
    scale_sq = scale_rmsse(train_wide, M)

    rows = []
    for col, label in METHODS.items():
        yhat = Y_rec_df.pivot(index="unique_id", columns="ds", values=col).loc[ids, test_wide.columns]
        rows.append(pd.DataFrame({
            "method": label,
            "unique_id": ids,
            "scale": scale,
            "mase": mase(test_wide.to_numpy(), yhat.to_numpy(), scale),
            "rmsse": rmsse(test_wide.to_numpy(), yhat.to_numpy(), scale_sq),
        }))
    metric_df = pd.concat(rows, ignore_index=True)

    tags_all = {**tags, "All series": ids}
    lm = level_mean(metric_df, tags_all, value_cols=["mase", "rmsse"], by=["method"])
    timings_metrics = time.perf_counter() - t0

    labels = list(METHODS.values())
    levels = list(tags_all)
    counts = lm[lm.method == "Base"].set_index("level").loc[levels, ["n_series", "n_excluded"]]

    def table(value):
        return lm.pivot(index="level", columns="method", values=value).loc[levels, labels]

    def show(tab, title):
        best = tab.idxmin(axis=1)
        out = tab.map(lambda x: f"{x:.4f}")
        for lvl in tab.index:
            out.loc[lvl, best[lvl]] += "*"
        out.insert(0, "excl", counts["n_excluded"])
        out.insert(0, "n", counts["n_series"])
        print(title + "   (* = best in row)")
        print(out.to_string())
        print()

    mase_tab, rmsse_tab = table("mase"), table("rmsse")
    print()
    show(mase_tab, "Mean MASE by level")
    show(rmsse_tab, "Mean RMSSE by level")

    # ---- gate checks, reported as-is ----
    for name, tab in [("MASE", mase_tab), ("RMSSE", rmsse_tab)]:
        chk = pd.DataFrame({
            "MinT<=WLS-var": tab["MinT-shrink"] <= tab["WLS-var"],
            "MinT<=WLS-struct": tab["MinT-shrink"] <= tab["WLS-struct"],
            "WLS-var<=OLS": tab["WLS-var"] <= tab["OLS"],
            "WLS-struct<=OLS": tab["WLS-struct"] <= tab["OLS"],
            "OLS<=Base": tab["OLS"] <= tab["Base"],
            "MinT<=Base": tab["MinT-shrink"] <= tab["Base"],
            "BU<MinT": tab["BottomUp"] < tab["MinT-shrink"],
            "gain_MinT_vs_Base": (tab["MinT-shrink"] - tab["Base"]) / tab["Base"],
            "gap_Base-MinT": tab["Base"] - tab["MinT-shrink"],
            "spread_max-min": tab.max(axis=1) - tab.min(axis=1),
        })
        print(f"Ordering checks ({name})")
        print(chk.to_string(float_format=lambda x: f"{x:+.4f}"))
        print()

    print("Runtime (seconds):")
    for stage, sec in timings.items():
        print(f"  {stage:<16} {sec:8.1f}   (measured when the cache was written)")
    print(f"  {'metrics':<16} {timings_metrics:8.1f}")


if __name__ == "__main__":
    main()
