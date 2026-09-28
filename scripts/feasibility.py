"""Weeks 4-5, step 1: can each model be run on this machine, and at what cost?

For every model in src/runners.py except chronos_bolt_small (already run):
  1. a short probe on 64 series, on CPU and on MPS, to compare speed and output;
  2. one full TourismLarge origin on the chosen device: the base forecast and
     the 24 backtest calls, both cached like any other forecast;
  3. checks of the base forecast only: shape, NaN, quantile crossings, negatives.

Nothing is scored. The test window is never loaded and no accuracy metric is
computed. Writes docs/feasibility.md.

Do not import hierarchicalforecast here (see src/runners.py).

Run from the repo root: python scripts/feasibility.py [model ...]
                        python scripts/feasibility.py --report   (rewrite the report only)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_hierarchy, rolling_origins  # noqa: E402
from src.forecasters import cache_paths, forecast, forecast_cutoffs, samples_path  # noqa: E402
from src.residuals import backtest_paths, inner_forecasts, inner_origins  # noqa: E402
from src.runners import MODELS, NUM_SAMPLES, QUANTILE_COLS, SEED, quantile_col  # noqa: E402

DATASET = "TourismLarge"
K, N_INNER = 5, 24
DEVICE = "cpu"              # the device used for the study; MPS is only probed
PROBE_SERIES = 64
LIMIT_HOURS = 2.0
OUT_JSON = ROOT / "docs" / "feasibility.json"
OUT_MD = ROOT / "docs" / "feasibility.md"
DEFAULT_MODELS = ["chronos_bolt_tiny", "chronos_bolt_mini", "chronos_t5_small", "chronos_2",
                  "tirex", "moirai_2_small"]
# Models whose 24 backtest calls are projected from 2 timed calls instead of run in
# full, because one origin alone takes over an hour. Their backtests are not cached.
PROJECT_BACKTEST = {"chronos_t5_small": 2}


def versions(env):
    """Versions of the packages that matter, read in the env that runs the model."""
    import subprocess
    code = ("import importlib.metadata as m, json, platform; "
            "names=['torch','numpy','pandas','chronos-forecasting','tirex-ts','uni2ts','transformers']; "
            "out={'python': platform.python_version()}; "
            "[out.update({n: m.version(n)}) for n in names if any(d.metadata['Name'].lower()==n for d in m.distributions())]; "
            "print(json.dumps(out))")
    import os
    import shutil
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    p = subprocess.run([conda, "run", "-n", env, "python", "-c", code], capture_output=True, text=True)
    return json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {"error": p.stderr[-300:]}


def probe(model, train, origin, h, freq):
    """Seconds for one call on a sample of series, per device, and how far the
    MPS forecast is from the CPU one."""
    ids = list(dict.fromkeys(train["unique_id"]))
    step = max(1, len(ids) // PROBE_SERIES)
    sub = train[train["unique_id"].isin(ids[::step][:PROBE_SERIES])]
    out, frames = {}, {}
    for device in ["cpu", "mps"]:
        try:
            forecast_cutoffs(model, sub, [origin], h, freq=freq, device=device)          # warm-up
            f, info = forecast_cutoffs(model, sub, [origin], h, freq=freq, device=device)
            out[device] = {"seconds": info["seconds"][0], "ok": True}
            frames[device] = f
        except Exception as e:                                                           # noqa: BLE001
            out[device] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[-300:]}"}
    if len(frames) == 2:
        a, b = frames["cpu"][QUANTILE_COLS].to_numpy(), frames["mps"][QUANTILE_COLS].to_numpy()
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1.0)
        out["mps_vs_cpu_max_rel_diff"] = float(rel.max())
        out["mps_vs_cpu_median_rel_diff"] = float(np.median(rel))
    return out


def check_base(base, spec, info_levels, n, h, samples=None):
    """Sanity checks of a base forecast. No actuals are involved."""
    native = [quantile_col(q) for q in info_levels] if info_levels else QUANTILE_COLS
    q = base[native].to_numpy()
    q9 = base[QUANTILE_COLS].to_numpy()
    width = q9[:, -1] - q9[:, 0]
    out = {
        "rows": int(len(base)), "rows_expected": n * h,
        "series": int(base["unique_id"].nunique()), "horizons": int(base["ds"].nunique()),
        "columns": list(base.columns),
        "first_ds": str(base["ds"].min().date()), "last_ds": str(base["ds"].max().date()),
        "nan": int(base[["yhat", "mean", *native]].isna().sum().sum()),
        "inf": int((~np.isfinite(base[["yhat", "mean", *native]].to_numpy())).sum()),
        "quantile_crossings_native": int((np.diff(q, axis=1) < 0).any(axis=1).sum()),
        "quantile_crossings_q10_q90": int((np.diff(q9, axis=1) < 0).any(axis=1).sum()),
        "q90_le_q10": int((width <= 0).sum()),
        "min_width_q90_q10": float(width.min()),
        "negative_median": int((base["yhat"] < 0).sum()),
        "negative_mean": int((base["mean"] < 0).sum()),
        "negative_q10": int((base["q10"] < 0).sum()),
        "median_is_q50": bool(np.allclose(base["yhat"], base["q50"])) if spec["output"] == "quantiles" else None,
        "mean_above_median_share": float((base["mean"] > base["yhat"]).mean()),
        "cells": n * h,
    }
    if samples is not None:
        s = samples[[c for c in samples.columns if c.startswith("s")]].to_numpy(dtype=np.float64)
        out.update({"samples_shape": [int(s.shape[0]), int(s.shape[1])], "samples_nan": int(np.isnan(s).sum()),
                    "samples_negative_share": float((s < 0).mean()),
                    "zero_sample_variance": int((s.var(axis=1, ddof=1) == 0).sum())})
    return out


def problems(rec):
    """Problems of a model, from its checks, timing and probe."""
    out = []
    c, t = rec.get("checks"), rec.get("timing")
    if "failed" in rec:
        out.append("failed to run")
    if c:
        if c["rows"] != c["rows_expected"]:
            out.append("wrong shape")
        if c["nan"] or c["inf"]:
            out.append(f"{c['nan']} NaN, {c['inf']} inf")
        if rec["output"] == "samples":
            if c.get("zero_sample_variance"):
                out.append(f"{c['zero_sample_variance']} forecasts with zero sample variance and "
                           f"{c['q90_le_q10']} with q90 = q10 (W_pv undefined)")
        elif c["q90_le_q10"]:
            out.append(f"{c['q90_le_q10']} forecasts with q90 <= q10 (W_pv undefined)")
    if t and t["projected_total_seconds"] > LIMIT_HOURS * 3600:
        out.append(f"projected over {LIMIT_HOURS:g} hours")
    p = rec.get("probe", {})
    if p and not p.get("mps", {}).get("ok", False):
        out.append("does not run on MPS")
    elif p.get("mps_vs_cpu_max_rel_diff", 0) > 1e-3 and rec["output"] == "quantiles":
        out.append(f"MPS forecasts differ from CPU by up to {p['mps_vs_cpu_max_rel_diff']:.1%}")
    return out


def run_model(model, train, origin, h, freq, n):
    spec = MODELS[model]
    rec = {"model": model, "env": spec["env"], "repo": spec["repo"], "revision": spec["revision"],
           "output": spec["output"], "device": DEVICE, "problems": []}
    try:
        rec["versions"] = versions(spec["env"])
        rec["probe"] = probe(model, train, origin, h, freq)

        base_path, timing_path = cache_paths(DATASET, origin, model)
        base_cached = base_path.exists()
        base = forecast(model, train, h, freq=freq, device=DEVICE, dataset=DATASET)
        bt_cached = backtest_paths(DATASET, origin, model)["inner"].exists()
        t_base = pd.read_parquet(timing_path).iloc[0]
        if model in PROJECT_BACKTEST and not bt_cached:
            inner_all = inner_origins(sorted(train["ds"].unique()), h, N_INNER)
            timed = [inner_all[0], inner_all[-1]][: PROJECT_BACKTEST[model]]   # shortest and longest history
            inner, info = forecast_cutoffs(model, train, timed, h, freq=freq, device=DEVICE)
            t_bt = {"seconds": float(np.mean(info["seconds"])) * N_INNER, "forecast_calls": N_INNER,
                    "load_seconds": info["load_seconds"]}
            rec["backtest_projected_from_calls"] = [float(x) for x in info["seconds"]]
        else:
            inner = inner_forecasts(model, train, h, N_INNER, freq=freq, device=DEVICE, dataset=DATASET)
            t_bt = pd.read_parquet(backtest_paths(DATASET, origin, model)["timing"]).iloc[0]

        levels = [float(c[1:]) / 100 for c in base.columns if c.startswith("q")] \
            if spec["output"] == "quantiles" else None
        rec["native_levels"] = sorted(levels) if levels else None
        rec["num_samples"] = NUM_SAMPLES if spec["output"] == "samples" else None
        sp = samples_path(DATASET, origin, model)
        samples = pd.read_parquet(sp) if sp.exists() else None
        rec["checks"] = check_base(base, spec, rec["native_levels"], n, h, samples)
        rec["backtest_rows"] = int(len(inner))
        rec["backtest_last_target"] = str(inner["ds"].max().date())
        rec["cached_before_this_run"] = {"base": base_cached, "backtest": bt_cached}

        load = float(max(t_base.get("load_seconds", 0.0), t_bt.get("load_seconds", 0.0)))
        per_origin = float(t_base["seconds"]) + float(t_bt["seconds"])
        rec["timing"] = {
            "base_seconds": float(t_base["seconds"]), "backtest_seconds": float(t_bt["seconds"]),
            "backtest_calls": int(t_bt["forecast_calls"]), "load_seconds": load,
            "seconds_per_origin": per_origin,
            "projected_total_seconds": K * per_origin + K * 2 * load,
        }
    except Exception as e:                                                               # noqa: BLE001
        rec["failed"] = f"{type(e).__name__}: {str(e)[-600:]}"
    return finish(rec)


def finish(rec):
    rec["problems"] = problems(rec)
    rec["drop_candidate"] = bool("failed" in rec or any("projected over" in p for p in rec["problems"]))
    return rec


def refresh_checks(rec, origin, n, h):
    """Recompute the checks of a record from the cached base forecast."""
    base_path, _ = cache_paths(DATASET, origin, rec["model"])
    if base_path.exists():
        sp = samples_path(DATASET, origin, rec["model"])
        rec["checks"] = check_base(pd.read_parquet(base_path), MODELS[rec["model"]], rec.get("native_levels"),
                                   n, h, pd.read_parquet(sp) if sp.exists() else None)
    return finish(rec)


def fmt_levels(rec):
    if rec["output"] == "samples":
        return f"none ({rec.get('num_samples')} sample paths)"
    lv = rec.get("native_levels") or []
    if len(lv) == 9:
        return "9: 0.1 to 0.9"
    return f"{len(lv)}: " + ", ".join(f"{x:g}" for x in lv)


def write_markdown(records, origin, train_len):
    def cell(x):
        return "n/a" if x is None else x

    rows = ["| Model | Env | Device | Output type | Native quantile levels | Seconds per origin | "
            "Projected total, 5 origins | Problems | Drop candidate |", "|" + "---|" * 9]
    for r in records:
        t = r.get("timing")
        rows.append("| " + " | ".join(str(cell(x)) for x in [
            r["model"], r["env"], r["device"],
            "sample paths" if r["output"] == "samples" else "quantiles", fmt_levels(r),
            f"{t['seconds_per_origin']:.1f}" if t else "n/a",
            f"{t['projected_total_seconds'] / 60:.1f} min" if t else "n/a",
            "; ".join(r["problems"]) or "none", "yes" if r["drop_candidate"] else "no"]) + " |")

    chk = ["| Model | Rows | NaN | Crossings, native levels | Crossings, q10 to q90 | q90 <= q10 | "
           "Negative medians | Negative means | Negative q10 |", "|" + "---|" * 9]
    prb = ["| Model | CPU, s | MPS, s | MPS against CPU, largest relative difference |", "|---|---|---|---|"]
    tim = ["| Model | Base forecast, s | 24 backtest calls, s | Model load, s |", "|---|---|---|---|"]
    ckp = ["| Model | Repository | Revision | Weight file | Weights last changed | SHA-256 of the weights |",
           "|---|---|---|---|---|---|"]
    ver = ["| Model | Env | Packages |", "|---|---|---|"]
    for r in records:
        c, p, t = r.get("checks"), r.get("probe", {}), r.get("timing")
        if c:
            chk.append(f"| {r['model']} | {c['rows']} of {c['rows_expected']} | {c['nan']} | "
                       f"{c['quantile_crossings_native']} | {c['quantile_crossings_q10_q90']} | "
                       f"{c['q90_le_q10']} | {c['negative_median']} | {c['negative_mean']} | "
                       f"{c['negative_q10']} |")
        cpu, mps = p.get("cpu", {}), p.get("mps", {})
        prb.append(f"| {r['model']} | {cpu.get('seconds', float('nan')):.2f} | "
                   + (f"{mps['seconds']:.2f}" if mps.get("ok") else "fails") + " | "
                   + (f"{p['mps_vs_cpu_max_rel_diff']:.1e}" if "mps_vs_cpu_max_rel_diff" in p else "n/a") + " |")
        if t:
            note = " (projected)" if "backtest_projected_from_calls" in r else ""
            tim.append(f"| {r['model']} | {t['base_seconds']:.1f} | {t['backtest_seconds']:.1f}{note} | "
                       f"{t['load_seconds']:.1f} |")
        m = MODELS[r["model"]]
        ckp.append(f"| {r['model']} | `{m['repo']}` | `{m['revision']}` | {m['weights_file']} | "
                   f"{m['weights_date']} | `{m['weights_sha256']}` |")
        ver.append(f"| {r['model']} | {r['env']} | "
                   + ", ".join(f"{k} {v}" for k, v in r.get("versions", {}).items()) + " |")

    text = f"""# Feasibility of the foundation models on this machine

Generated by `scripts/feasibility.py`. Do not edit by hand.

Dataset {DATASET}, outer origin {origin.date()} ({train_len} months of history), h = 12, 555 series.
One origin costs one base forecast and {N_INNER} backtest calls. Each call forecasts all 555 series.
Nothing was scored: the test window was not loaded and no accuracy metric was computed.

## Summary

{chr(10).join(rows)}

The projected total is 5 times the seconds per origin, plus the model load time. A model is a drop
candidate if it fails to run or is projected over {LIMIT_HOURS:g} hours. The study device is {DEVICE.upper()}.

## Time per origin

{chr(10).join(tim)}

The model load time was measured only for chronos_t5_small and moirai_2_small. For the others the
model was already in memory when it was timed, and 0.0 is shown.

"Projected" means the {N_INNER} backtest calls were not all run: two were timed, from the earliest and
the latest inner origin, and their mean was multiplied by {N_INNER}. Those backtests are not cached.

## CPU against MPS, probe on {PROBE_SERIES} series, one call

{chr(10).join(prb)}

## Checks of the base forecast

{chr(10).join(chk)}

Counts are over 555 series and 12 horizons (6,660 forecasts). A crossing is a forecast whose quantiles
are not in increasing order.

## Checkpoints

{chr(10).join(ckp)}

tirex is the original checkpoint. `NX-AI/TiRex-1.1-gifteval` was trained on a different corpus
(docs/leakage.md, section 2.4) and is not used.

## Output types and the two point arms

| Model | Output | Median arm | Mean arm |
|---|---|---|---|
| chronos_bolt_tiny, chronos_bolt_mini, chronos_bolt_small | 9 quantiles, 0.1 to 0.9 | the 0.5 quantile | average of the 9 quantiles |
| chronos_t5_small | {NUM_SAMPLES} sample paths, seed fixed per call | sample median | sample mean |
| chronos_2 | 21 quantiles: 0.01, 0.05, 0.1, then steps of 0.05 to 0.9, 0.95, 0.99 | the 0.5 quantile | average of the 21 quantiles |
| tirex | 9 quantiles, 0.1 to 0.9 | the 0.5 quantile | average of the 9 quantiles |
| moirai_2_small | 9 quantiles, 0.1 to 0.9 | the 0.5 quantile | average of the 9 quantiles |

- No quantile model supplies a mean of its own. Chronos-Bolt, Chronos-2 and TiRex return the median
  as their "mean" output (checked), and Moirai-2 returns quantiles only.
- q10 to q90 are stored for every model. For chronos_t5_small they are sample quantiles. The other
  native levels of chronos_2 (q1, q5, q15, ..., q95, q99) are stored as extra columns.
- PREREG §5 defines the predictive variance as the sample variance for sample-path models and from
  q10 and q90 otherwise. The check "q90 <= q10" below is reported for every model; for
  chronos_t5_small the sample variance is the registered quantity.
- chronos_t5_small: the seed of a call is {SEED} plus the day number of its cutoff date, so every call
  is reproducible on its own. The sample paths of base forecasts are stored; those of backtest calls are not.
- moirai_2_small: uni2ts has no default context length, it is a required argument. It is set to the
  length of the history given, so nothing is cut off or padded.

## Chronos-2 modes

Run here: univariate. Each series is a separate task and `cross_learning=False` (the default), so no
information passes between series.

Available and not run:

| Mode | How it is switched on | Note from the library documentation |
|---|---|---|
| Cross-learning | `cross_learning=True` | All series in a batch are predicted jointly. Results depend on the batch size; about 100 is recommended. |
| Multivariate | a task with several target rows | Information is shared among the variates of one task. |
| Covariates | `past_covariates`, `future_covariates` | Past-only and known-future covariates per task. |

## Versions

{chr(10).join(ver)}
"""
    OUT_MD.write_text(text)


def main():
    report_only = "--report" in sys.argv
    models = [] if report_only else ([a for a in sys.argv[1:]] or DEFAULT_MODELS)
    Y_df, S_df, tags, freq, h = load_hierarchy(DATASET)
    split = rolling_origins(Y_df, h, K)[-1]            # the latest origin has the longest history
    origin, train = split["origin"], split["train"]    # split["test"] is not used
    n = train["unique_id"].nunique()
    train_len = train["ds"].nunique()

    records = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else []
    records = [refresh_checks(r, origin, n, h) for r in records if r["model"] not in models]
    for model in models:
        t0 = time.perf_counter()
        rec = run_model(model, train, origin, h, freq, n)
        rec["wall_seconds_this_run"] = time.perf_counter() - t0
        records.append(rec)
        t = rec.get("timing")
        print(f"{model}: " + (f"{t['seconds_per_origin']:.1f}s per origin, projected "
                              f"{t['projected_total_seconds'] / 60:.1f} min" if t else rec.get("failed", ""))
              + f"; problems: {'; '.join(rec['problems']) or 'none'}", flush=True)
        OUT_JSON.write_text(json.dumps(records, indent=1, default=str))

    order = {m: i for i, m in enumerate(DEFAULT_MODELS)}
    records.sort(key=lambda r: order.get(r["model"], 99))
    OUT_JSON.write_text(json.dumps(records, indent=1, default=str))
    write_markdown(records, origin, train_len)
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
