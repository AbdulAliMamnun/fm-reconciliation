"""Week 3: the six residual-free and backtest W-estimators (H2) for
Chronos-Bolt-small on TourismLarge, 5 origins, both point arms.

    OLS, WLS-struct             need nothing
    WLS-pv                      the model's own predictive variance
    WLS-var_bt, MinT-shrink_bt  24 h-step backtest residuals per horizon
    Hybrid                      predictive variance + shrunk backtest correlations

W is built per horizon. The median arm is the registered one. The mean arm
(average of the 9 quantiles) is EXPLORATORY for Chronos-Bolt x TourismLarge.
Descriptive only: no bootstrap, no Model Confidence Set.

Everything is appended to results/results.parquet and the accuracy tables are
computed from that table. The AutoETS rows written by notebooks/02_gate_ets.py
are reference columns.

Do not import hierarchicalforecast here (see src/forecasters.py).

Run from the repo root: python notebooks/05_h2_chronos_bolt.py [--per-origin]
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
    W_hybrid, W_ols, W_pv, W_shrink_bt, W_struct, W_var_bt, predictive_variance, projection_matrix,
    reconcile, reconcile_by_horizon,
)
from src.residuals import backtest_paths, backtest_residuals, residual_array  # noqa: E402
from src.results import append_results, build_results  # noqa: E402

DATASET = "TourismLarge"
MODEL = "chronos_bolt_small"
DEVICE = "cpu"
K = 5
N_INNER = 24
COHERENCE_TOL = 1e-8
TABLE = ROOT / "results" / "results.parquet"

POINTS = ["median", "mean"]
W_LABEL = {
    "base": "Base", "ols": "OLS", "wls_struct": "WLS-struct", "wls_pv": "WLS-pv",
    "wls_var_bt": "WLS-var_bt", "mint_shrink_bt": "MinT-shrink_bt", "hybrid": "Hybrid",
}
W_ESTS = list(W_LABEL)
NEEDS_BACKTEST = {"wls_var_bt", "mint_shrink_bt", "hybrid"}
ETS = {("AutoETS", "mean", "base"): "ETS Base", ("AutoETS", "mean", "mint_shrink"): "ETS MinT-shrink"}
EXPLORATORY = "EXPLORATORY: mean arm, post hoc for Chronos-Bolt x TourismLarge"


def mark_best(tab, own, counts=None):
    """+ marks the best Chronos-Bolt column, * the best of all columns."""
    out = tab.map(lambda x: f"{x:.4f}")
    best, best_own = tab.idxmin(axis=1), tab[own].idxmin(axis=1)
    for idx in tab.index:
        out.loc[idx, best_own[idx]] += "+"
        out.loc[idx, best[idx]] += "*"
    if counts is not None:
        out.insert(0, "n", counts["n_series"])
    return out


def main():
    per_origin = "--per-origin" in sys.argv
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    t_start = time.perf_counter()

    Y_df, S_df, tags, freq, h = load_hierarchy(DATASET)
    m = DATASETS[DATASET]["m"]
    ids = S_df["unique_id"].to_numpy()
    bottom_ids = [c for c in S_df.columns if c != "unique_id"]
    S = S_df[bottom_ids].to_numpy(dtype=np.float64)
    n, nb = S.shape
    level_order = list(tags)
    level_of = pd.Series({u: lvl for lvl, us in tags.items() for u in us}).loc[ids].to_numpy()
    bottom_level = level_order[-1]
    wide = Y_df.pivot(index="unique_id", columns="ds", values="y").loc[ids]

    print(f"{DATASET}: {n} series, {K} origins, h={h}, model {MODEL}, device {DEVICE}, "
          f"{N_INNER} inner origins per outer origin")
    print(f"arms: median (registered, primary) and mean. {EXPLORATORY}")

    new_rows, cost_rows, lam_rows, weight_rows, ratio_rows, run_rows = [], [], [], [], [], []
    self_rows = []
    n_under = S.sum(axis=1)          # number of bottom series under each series
    for split in rolling_origins(Y_df, h, K):
        origin, train, test = split["origin"], split["train"], split["test"]
        base_cached = cache_paths(DATASET, origin, MODEL)[0].exists()
        bt_paths = backtest_paths(DATASET, origin, MODEL)
        bt_cached = bt_paths["inner"].exists()

        base = forecast(MODEL, train, h, freq=freq, device=DEVICE, dataset=DATASET)
        base_timing = pd.read_parquet(cache_paths(DATASET, origin, MODEL)[1]).iloc[0]
        dates = np.sort(base["ds"].unique())

        def to_wide(values):
            return (base.assign(_v=values).pivot(index="unique_id", columns="ds", values="_v")
                    .loc[ids].to_numpy())

        q = np.stack([to_wide(base[c].to_numpy()) for c in QUANTILE_COLS], axis=-1)   # (n, h, 9)
        v_pv = predictive_variance(q)                                                 # (n, h)

        y_train = wide.loc[:, :origin].to_numpy()
        scales = pd.DataFrame({"unique_id": ids, "mase_scale": scale_mase(y_train, m),
                               "rmsse_scale": scale_rmsse(y_train, m)})
        quant = base[["unique_id", "ds", *QUANTILE_COLS]].assign(W_est="base")

        for point in POINTS:
            resid_df = backtest_residuals(MODEL, train, h, N_INNER, point=point, freq=freq,
                                          device=DEVICE, dataset=DATASET)
            if resid_df["ds"].max() > origin:
                raise RuntimeError("A backtest residual target lies after the outer origin.")
            resid = residual_array(resid_df, ids)                                     # (n, 24, h)
            bt_timing = pd.read_parquet(bt_paths["timing"]).iloc[0]
            y_hat = to_wide(point_forecast(base, point))

            rec, build_s, rec_s, built = {"base": y_hat}, {}, {}, {}

            def run(name, build, per_horizon=True):
                t0 = time.perf_counter()
                W = build()
                build_s[name] = time.perf_counter() - t0
                built[name] = W if per_horizon else [W]
                t0 = time.perf_counter()
                rec[name] = reconcile_by_horizon(y_hat, S, W) if per_horizon else reconcile(y_hat, S, W)
                rec_s[name] = time.perf_counter() - t0
                return W

            run("ols", lambda: W_ols(n), per_horizon=False)
            run("wls_struct", lambda: W_struct(S), per_horizon=False)
            run("wls_pv", lambda: W_pv(q))
            Wvar = run("wls_var_bt", lambda: W_var_bt(resid))
            lams = []

            def build_shrink():
                Ws, lam_h = W_shrink_bt(resid, return_lambda=True)
                lams.extend(lam_h)
                return Ws

            run("mint_shrink_bt", build_shrink)
            run("hybrid", lambda: W_hybrid(q, resid))

            fc = pd.DataFrame({"unique_id": np.repeat(ids, h), "ds": np.tile(dates, n)})
            for w in W_ESTS:
                fc[w] = rec[w].reshape(-1)
            new_rows.append(build_results(DATASET, origin, MODEL, fc, test, tags, scales,
                                          {w: w for w in W_ESTS}, point=point, quantiles=quant))

            for w in W_ESTS[1:]:
                extra = N_INNER if w in NEEDS_BACKTEST else 0
                cost_rows.append({
                    "origin": origin.date(), "point": point, "W_est": W_LABEL[w],
                    "extra_forecast_calls": extra,
                    "extra_series_forecasts": extra * n,
                    "backtest_forecast_s": float(bt_timing["seconds"]) if extra else 0.0,
                    "build_W_s": build_s[w], "reconcile_s": rec_s[w],
                })
            # How the reconciled Country forecast is composed. It is c'y_hat with c the
            # first row of S G. Because S G S = S, the weights satisfy c'S = 1', so
            # sum_i c_i * (bottom series under i) / n_bottom = 1. Grouped by level, these
            # terms say how much of the reconciled total comes from each level's forecasts.
            for w, Ws in built.items():
                for t, W in enumerate(Ws):
                    c = S[0] @ projection_matrix(S, W)
                    part = pd.Series(c * n_under / nb).groupby(level_of).sum()
                    for lvl in level_order:
                        self_rows.append({"origin": origin.date(), "point": point,
                                          "W_est": W_LABEL[w], "horizon": t + 1 if len(Ws) > 1 else 0,
                                          "level": lvl, "share": part[lvl]})
            for t in range(h):
                lam_rows.append({"origin": origin.date(), "point": point, "horizon": t + 1,
                                 "lambda": lams[t]})
                v_bt = np.diag(Wvar[t])
                for name, var in [("WLS-pv", v_pv[:, t]), ("WLS-var_bt", v_bt)]:
                    wgt = pd.Series(1.0 / var).groupby(level_of).sum()
                    for lvl in level_order:
                        weight_rows.append({"origin": origin.date(), "point": point, "horizon": t + 1,
                                            "W_est": name, "level": lvl, "share": wgt[lvl] / wgt.sum()})
                ratio_rows.append(pd.DataFrame({
                    "origin": origin.date(), "point": point, "horizon": t + 1, "level": level_of,
                    "series_id": ids, "v_pv": v_pv[:, t], "v_bt": v_bt,
                    "bt_bias": resid[:, :, t].mean(axis=1), "bt_var": resid[:, :, t].var(axis=1, ddof=1),
                    "mase_scale": scales["mase_scale"].to_numpy(),
                }))

        run_rows.append({
            "origin": origin.date(), "train_len": y_train.shape[1],
            "base": "cache" if base_cached else "computed",
            "backtest": "cache" if bt_cached else "computed",
            "base_forecast_s": float(base_timing["seconds"]),
            "backtest_forecast_s": float(bt_timing["seconds"]),
            "backtest_calls": int(bt_timing["forecast_calls"]),
            "inner_from": resid_df["inner_origin"].min().date(),
            "inner_to": resid_df["inner_origin"].max().date(),
            "last_target": resid_df["ds"].max().date(),
        })
        print(f"  origin {origin.date()}: base {run_rows[-1]['base']}, backtest "
              f"{run_rows[-1]['backtest']} ({run_rows[-1]['backtest_calls']} calls, "
              f"{run_rows[-1]['backtest_forecast_s']:.1f}s)", flush=True)

    append_results(TABLE, pd.concat(new_rows, ignore_index=True))

    runs = pd.DataFrame(run_rows)
    print("\nBacktest design check (no target after the outer origin):")
    print(runs[["origin", "train_len", "inner_from", "inner_to", "last_target", "backtest_calls"]]
          .to_string(index=False))
    assert (pd.to_datetime(runs["last_target"]) <= pd.to_datetime(runs["origin"])).all()

    # ---- accuracy: everything below reads only the results table ----
    res_all = pd.read_parquet(TABLE)
    res_all = res_all[res_all["dataset"] == DATASET]
    tags_tab = {**{k2: np.asarray(v2) for k2, v2 in tags.items()}, "All series": ids}
    levels = list(tags_tab)
    top = levels[0]
    legend = "(+ = best Chronos-Bolt column, * = best of all columns)"

    cb = res_all[res_all.model == MODEL]
    coh = []
    for (origin, point, w), g in cb.groupby(["origin", "point", "W_est"], sort=False):
        if w not in W_LABEL:
            continue
        yh = g.pivot(index="series_id", columns="horizon", values="yhat")
        coh.append({"origin": pd.Timestamp(origin).date(), "point": point, "W_est": W_LABEL[w],
                    "max_gap": np.abs(S @ yh.loc[bottom_ids].to_numpy() - yh.loc[ids].to_numpy()).max(),
                    "n_negative": int((yh.to_numpy() < 0).sum())})
    coh = pd.DataFrame(coh)
    worst = coh[coh.W_est != "Base"].groupby("W_est", sort=False)["max_gap"].max()
    print(f"\nCoherence: worst max |S b - y| over origins and arms, by W (tolerance {COHERENCE_TOL:g})")
    print(worst.to_frame().T.to_string(float_format=lambda x: f"{x:.2e}", index=False))
    print(f"all reconciled forecasts coherent: {'YES' if worst.max() <= COHERENCE_TOL else 'NO'}")
    print("\nNegative point forecasts, total over 5 origins (of "
          f"{n * h * K} per column):")
    print(coh.pivot_table(index="point", columns="W_est", values="n_negative", aggfunc="sum")
          .loc[POINTS, list(W_LABEL.values())].to_string())

    tables = {}
    for point in POINTS:
        cols = {(MODEL, point, w): W_LABEL[w] for w in W_ESTS} | ETS
        own = [W_LABEL[w] for w in W_ESTS]
        labels = list(cols.values())
        keys = pd.Series(list(zip(res_all["model"], res_all["point"], res_all["W_est"])), index=res_all.index)
        res = res_all[keys.isin(list(cols))].copy()
        missing = set(cols) - set(keys)
        if missing:
            raise SystemExit(f"Missing from the results table: {missing}.")
        sm = series_metrics(res)
        sm["col"] = [cols[k2] for k2 in zip(sm["model"], sm["point"], sm["W_est"])]
        lm = level_mean(sm, tags_tab, value_cols=["mase", "rmsse"], by=["origin", "col"],
                        scale_col="mase_scale", id_col="series_id")
        counts = (lm[lm.col == "Base"].groupby("level")
                  .agg(n_series=("n_series", "first"), n_excluded=("n_excluded", "sum")).loc[levels])
        avg = lm.groupby(["level", "col"], sort=False)[["mase", "rmsse"]].mean().reset_index()
        title = f"ARM: {point}" + (f"   [{EXPLORATORY}]" if point == "mean" else "   [registered, primary]")
        print("\n" + "=" * 110 + f"\n{title}\n" + "=" * 110)
        print(f"series excluded by the scale < 1e-8 rule: {int(counts.loc['All series', 'n_excluded'])}")
        for value in ["mase", "rmsse"]:
            tab = avg.pivot(index="level", columns="col", values=value).loc[levels, labels]
            tables[(point, value)] = tab
            print(f"\nMean {value.upper()} by level, {point} arm, averaged over {K} origins   {legend}")
            print(mark_best(tab, own, counts).to_string())
            gain = pd.DataFrame({c: (tab[c] - tab["Base"]) / tab["Base"] for c in own[1:]})
            print(f"\nRelative change in {value.upper()} against Base of the same arm, negative = better")
            print((100 * gain).to_string(float_format=lambda x: f"{x:+.2f}%"))

        lm_top = lm[lm.level == top].assign(origin=lambda d: d["origin"].dt.date)
        tab = lm_top.pivot(index="origin", columns="col", values="mase")[labels]
        tab.loc["mean over origins"] = tab.mean()
        print(f"\n{top} MASE by origin, {point} arm   {legend}")
        print(mark_best(tab, own).to_string())

        bias = sm[sm.level == top].assign(origin=lambda d: d["origin"].dt.date)
        bias_tab = bias.pivot(index="origin", columns="col", values="bias")[labels]
        mean_y = (res[(res.level == top) & (res.W_est == "base") & (res.model == MODEL)]
                  .groupby("origin")["y"].mean().rename(lambda o: pd.Timestamp(o).date()))
        out = bias_tab.copy()
        out.insert(0, "mean_y", mean_y)
        print(f"\n{top} bias by origin, {point} arm: mean over the 12 horizons of (yhat - y)")
        print(out.to_string(float_format=lambda x: f"{x:.0f}"))
        print(f"\n{top} bias as % of the mean actual, {point} arm")
        print((100 * bias_tab.div(mean_y, axis=0)).to_string(float_format=lambda x: f"{x:+.2f}%"))

        if per_origin:
            for origin in sorted(lm["origin"].unique()):
                sub = lm[lm.origin == origin]
                t2 = sub.pivot(index="level", columns="col", values="mase").loc[levels, labels]
                print(f"\nOrigin {pd.Timestamp(origin).date()}, {point} arm: mean MASE by level   {legend}")
                print(mark_best(t2, own).to_string())

    # ---- lambda ----
    lam = pd.DataFrame(lam_rows)
    print("\n" + "=" * 110 + "\nShrinkage lambda (weight on the diagonal target), MinT-shrink_bt and "
          "Hybrid\n" + "=" * 110)
    print(f"one value per origin, horizon and arm: {N_INNER} residuals per series, {n} series")
    print(lam.groupby("point", sort=False)["lambda"].agg(["min", "median", "max", "count"])
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nby origin (min / median / max over the 12 horizons):")
    print(lam.groupby(["point", "origin"], sort=False)["lambda"].agg(["min", "median", "max"])
          .unstack("point").to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nby horizon (median over the 5 origins):")
    print(lam.pivot_table(index="point", columns="horizon", values="lambda", aggfunc="median")
          .loc[POINTS].to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"lambda == 1 (no correlation kept): {int((lam['lambda'] >= 1).sum())} of {len(lam)}; "
          f"lambda == 0: {int((lam['lambda'] <= 0).sum())}")

    # ---- inverse-variance weights ----
    wt = pd.DataFrame(weight_rows)
    print("\n" + "=" * 110 + "\nShare of the inverse-variance weight (sum of 1 / W_ii) by level\n" + "=" * 110)
    print("WLS-pv does not depend on the arm. Mean over origins and horizons.")
    share = (wt.groupby(["W_est", "point", "level"], sort=False)["share"].mean().unstack(["W_est", "point"])
             .loc[level_order])
    share = share[[("WLS-pv", "median"), ("WLS-var_bt", "median"), ("WLS-var_bt", "mean")]]
    print((100 * share).to_string(float_format=lambda x: f"{x:.3f}%"))
    bshare = wt[wt.level == bottom_level].groupby(["W_est", "point"], sort=False)["share"]
    print(f"\nbottom level ({bottom_level}), share over origins and horizons:")
    print((100 * bshare.agg(["min", "median", "max"])).to_string(float_format=lambda x: f"{x:.4f}%"))

    rt = pd.concat(ratio_rows, ignore_index=True)
    rt["ratio"] = rt["v_bt"] / rt["v_pv"]
    rt["bias_share"] = rt["bt_bias"] ** 2 / rt["v_bt"]
    rt["scaled_bias"] = rt["bt_bias"] / rt["mase_scale"]
    floor = rt[rt.v_bt <= 1e-7]
    print(f"\nseries x horizon x origin cells where WLS-var_bt is at its floor of 2e-8 (all "
          f"{N_INNER} residuals are 0):")
    print(floor.groupby("point").size().reindex(POINTS, fill_value=0).to_string()
          + f"   of {len(rt) // 2} per arm; distinct series: "
          f"{floor.groupby('point')['series_id'].nunique().reindex(POINTS, fill_value=0).to_dict()}")
    ok = rt[rt.v_bt > 1e-7]
    for point in POINTS:
        g = ok[ok.point == point].groupby("level", sort=False)
        d = pd.DataFrame({
            "median v_bt / v_pv": g["ratio"].median(),
            "share of v_bt that is bias^2 (median)": g["bias_share"].median(),
            "mean backtest bias / mase_scale": g["scaled_bias"].mean(),
        }).loc[level_order]
        print(f"\n{point} arm: backtest mean squared error (v_bt) against predictive variance (v_pv), "
              "by level")
        print(d.to_string(float_format=lambda x: f"{x:.3f}"))

    sw = pd.DataFrame(self_rows)
    print("\n" + "=" * 110 + "\nWhat the reconciled Country forecast is made of\n" + "=" * 110)
    print("Share of the reconciled Country forecast that comes from each level's base forecasts. "
          "The shares sum to 1. Unreconciled: Country = 1. BottomUp: bottom level = 1. "
          "Mean over origins and horizons.")
    for point in POINTS:
        comp = (sw[sw.point == point].groupby(["level", "W_est"], sort=False)["share"].mean()
                .unstack("W_est").loc[level_order, [W_LABEL[w] for w in W_ESTS[1:]]])
        comp.loc["sum"] = comp.sum()
        print(f"\n{point} arm" + ("" if point == "median" else f"   [{EXPLORATORY}]"))
        print(comp.to_string(float_format=lambda x: f"{x:.3f}"))

    # ---- cost ----
    cost = pd.DataFrame(cost_rows)
    print("\n" + "=" * 110 + "\nCompute cost per W (H2)\n" + "=" * 110)
    print(f"One forecast call = the model run once for all {n} series at one origin. Every W also "
          "needs the base forecast itself (1 call per origin), which is not counted below.")
    per = (cost.groupby(["W_est"], sort=False)
           .agg(extra_calls_per_origin=("extra_forecast_calls", "first"),
                extra_series_forecasts_per_origin=("extra_series_forecasts", "first"),
                backtest_forecast_s=("backtest_forecast_s", "mean"),
                build_W_s=("build_W_s", "mean"), reconcile_s=("reconcile_s", "mean")))
    per["total_s_per_origin"] = per[["backtest_forecast_s", "build_W_s", "reconcile_s"]].sum(axis=1)
    print("mean seconds per origin and arm (backtest forecasts are shared by the two arms and by the "
          "three backtest estimators):")
    print(per.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nForecast time per origin, seconds (measured when each cache was written):")
    print(runs[["origin", "base", "backtest", "base_forecast_s", "backtest_forecast_s", "backtest_calls"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"totals: base forecasts {runs.base_forecast_s.sum():.1f}s ({K} calls), backtest forecasts "
          f"{runs.backtest_forecast_s.sum():.1f}s ({int(runs.backtest_calls.sum())} calls); "
          f"this run wall clock {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
