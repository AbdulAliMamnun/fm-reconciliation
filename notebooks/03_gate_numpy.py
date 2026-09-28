"""Correctness gate, part 2 (PREREG §8.2): numpy reconciler vs hierarchicalforecast.

For each cached origin of TourismLarge / AutoETS, reconciles the cached base
forecasts with src/reconcile.py and compares with the cached
hierarchicalforecast output. Needs notebooks/02_gate_ets.py to have run.

Run from the repo root: python notebooks/03_gate_numpy.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hierarchicalforecast.methods import MinTrace  # noqa: E402

from src.data import load_hierarchy  # noqa: E402
from src.reconcile import W_ols, W_shrink, W_struct, W_var, projection_matrix, reconcile  # noqa: E402

DATASET, MODEL = "TourismLarge", "AutoETS"
TOL = 1e-8
METHODS = ["ols", "wls_struct", "wls_var", "mint_shrink"]
RESULTS = ROOT / "results"


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    _, S_df, _, _, _ = load_hierarchy(DATASET)
    ids = S_df["unique_id"].to_numpy()
    bottom_ids = [c for c in S_df.columns if c != "unique_id"]
    S = S_df[bottom_ids].to_numpy(dtype=np.float64)
    n, nb = S.shape

    rec_paths = sorted(RESULTS.glob(f"rec_{DATASET}_*_{MODEL}_hf.parquet"))
    if not rec_paths:
        raise SystemExit("No cached forecasts. Run notebooks/02_gate_ets.py first.")

    rows, lam_rows = [], []
    for rec_path in rec_paths:
        origin = rec_path.name.split("_")[2]
        rec = pd.read_parquet(rec_path)
        fitted = pd.read_parquet(RESULTS / f"fitted_{DATASET}_{origin}_{MODEL}.parquet")

        def wide(df, col):
            return df.pivot(index="unique_id", columns="ds", values=col).loc[ids].to_numpy(dtype=np.float64)

        y_hat = wide(rec, MODEL)
        y_in, y_fit = wide(fitted, "y"), wide(fitted, MODEL)
        resid = y_in - y_fit

        t0 = time.perf_counter()
        W_s, lam = W_shrink(resid, return_lambda=True)
        t_shrink = time.perf_counter() - t0
        Ws = {"ols": W_ols(n), "wls_struct": W_struct(S), "wls_var": W_var(resid), "mint_shrink": W_s}

        for m in METHODS:
            t0 = time.perf_counter()
            y_tilde = reconcile(y_hat, S, Ws[m])
            seconds = time.perf_counter() - t0
            hf_cached = wide(rec, f"{MODEL}/MinTrace_method-{m}")
            lib = MinTrace(method=m, nonnegative=False)
            hf_direct = lib.fit_predict(S=S, y_hat=y_hat, y_insample=y_in, y_hat_insample=y_fit)["mean"]
            G = projection_matrix(S, Ws[m])
            scale = np.sqrt(np.outer(np.diag(lib.W), np.diag(lib.W)))
            diff = np.abs(y_tilde - hf_cached)
            rows.append({
                "origin": origin, "method": m,
                "max_abs_diff": diff.max(),
                "max_rel_diff": (diff / np.maximum(np.abs(hf_cached), 1.0)).max(),
                "vs_direct_call": np.abs(y_tilde - hf_direct).max(),
                "W_diff_corr_scale": (np.abs(Ws[m] - lib.W) / scale).max(),
                "SGS_minus_S": np.abs(S @ G @ S - S).max(),
                "coherence": np.abs(S @ y_tilde[-nb:] - y_tilde).max(),
                "cond_W": np.linalg.cond(Ws[m]),
                "pass": bool(diff.max() <= TOL),
                "seconds": seconds,
            })
            if m == "mint_shrink":
                cov = np.cov(resid)
                low = np.tril_indices(n, k=-1)
                ratio = lib.W[low] / cov[low]
                lam_hf = 1.0 - np.median(ratio)
                lam_rows.append({
                    "origin": origin, "T": resid.shape[1],
                    "lambda_ours": lam, "lambda_hf": lam_hf, "abs_diff": abs(lam - lam_hf),
                    "hf_ratio_spread": np.abs(ratio - np.median(ratio)).max(),
                    "W_shrink_seconds": t_shrink,
                })

    res = pd.DataFrame(rows)
    fmt = {c: (lambda x: f"{x:.3e}") for c in
           ["max_abs_diff", "max_rel_diff", "vs_direct_call", "W_diff_corr_scale", "SGS_minus_S",
            "coherence", "cond_W"]}
    fmt["seconds"] = lambda x: f"{x:.3f}"

    print(f"{DATASET} / {MODEL}: numpy reconciler vs hierarchicalforecast, tolerance {TOL:g}")
    print(f"max |y_hat| = {np.abs(y_hat).max():.1f}\n")
    print("By origin and method:")
    print(res.to_string(index=False, formatters=fmt))

    worst = res.groupby("method", sort=False).agg(
        max_abs_diff=("max_abs_diff", "max"), max_rel_diff=("max_rel_diff", "max"),
        W_diff_corr_scale=("W_diff_corr_scale", "max"), SGS_minus_S=("SGS_minus_S", "max"),
        coherence=("coherence", "max"), origins_passing=("pass", "sum"), origins=("pass", "size"),
    )
    print("\nWorst case over origins, by method:")
    print(worst.to_string(formatters={k: v for k, v in fmt.items() if k in worst.columns}))

    print("\nShrinkage lambda (weight on the diagonal target; hierarchicalforecast's is backed out "
          "from its W):")
    print(pd.DataFrame(lam_rows).to_string(
        index=False,
        formatters={"lambda_ours": lambda x: f"{x:.15f}", "lambda_hf": lambda x: f"{x:.15f}",
                    "abs_diff": lambda x: f"{x:.3e}", "hf_ratio_spread": lambda x: f"{x:.3e}",
                    "W_shrink_seconds": lambda x: f"{x:.3f}"}))

    ok = bool(res["pass"].all())
    print(f"\nPREREG §8.2 (max abs diff <= {TOL:g} for every method and origin): "
          f"{'YES' if ok else 'NO'}; worst {res.max_abs_diff.max():.3e}")


if __name__ == "__main__":
    main()
