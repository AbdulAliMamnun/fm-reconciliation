"""Produce and cache the forecasts of one dataset: base forecasts and backtests.

Meant to run detached from the session:

    nohup python -u scripts/run_forecasts.py TourismLarge tirex chronos_2 \\
        > results/logs/forecasts.log 2>&1 &

Every finished piece is written to results/ at once: each base forecast, and
each of the 24 backtest calls of an origin (src/residuals.py). Anything already
in the cache is skipped, so the script can be stopped and started again at any
point and loses at most the forecast call that was in progress.

It forecasts and caches only. It reads no test window and scores nothing.
Progress is written to results/logs/forecasts_status.json.

Do not import hierarchicalforecast here (see src/runners.py).
"""
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_hierarchy, rolling_origins  # noqa: E402
from src.forecasters import cache_paths, forecast  # noqa: E402
from src.residuals import backtest_paths, inner_forecasts  # noqa: E402
from src.runners import MODELS  # noqa: E402

K, N_INNER = 5, 24
DEVICE = "cpu"                           # Deviations log: all models run on CPU
BASE_ONLY = {"chronos_t5_small"}         # Deviations log: no backtest residuals
LOGS = ROOT / "results" / "logs"
STATUS = LOGS / "forecasts_status.json"


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def write_status(status):
    status["updated"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=1))
    tmp.replace(STATUS)


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: run_forecasts.py DATASET MODEL [MODEL ...]")
    dataset, models = sys.argv[1], sys.argv[2:]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown models {unknown}")
    LOGS.mkdir(parents=True, exist_ok=True)

    Y_df, S_df, tags, freq, h = load_hierarchy(dataset)
    splits = rolling_origins(Y_df, h, K)
    status = {"dataset": dataset, "models": models, "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
              "state": "running", "jobs": {}}
    jobs = [(m, s) for m in models for s in splits]
    for m, s in jobs:
        status["jobs"][f"{m} {s['origin'].date()}"] = "pending"
    write_status(status)
    log(f"{dataset}: {len(models)} models x {len(splits)} origins, device {DEVICE}")

    failed = 0
    for model, split in jobs:
        origin, train = split["origin"], split["train"]          # split["test"] is never used
        name = f"{model} {origin.date()}"
        base_path, _ = cache_paths(dataset, origin, model)
        bt = backtest_paths(dataset, origin, model)
        need_bt = model not in BASE_ONLY
        if base_path.exists() and (not need_bt or bt["inner"].exists()):
            status["jobs"][name] = "done (was cached)"
            write_status(status)
            log(f"{name}: cached, skipped")
            continue
        status["jobs"][name] = "running"
        write_status(status)
        t0 = time.perf_counter()
        try:
            base_was_cached = base_path.exists()
            forecast(model, train, h, freq=freq, device=DEVICE, dataset=dataset)
            log(f"{name}: base forecast {'cached' if base_was_cached else 'written'}")
            if need_bt:
                n_parts = len(list(bt["parts"].glob("*.json"))) if bt["parts"].exists() else 0
                if n_parts:
                    log(f"{name}: continuing the backtests, {n_parts} of {N_INNER} calls already done")
                inner = inner_forecasts(model, train, h, N_INNER, freq=freq, device=DEVICE, dataset=dataset)
                if inner["ds"].max() > origin:
                    raise RuntimeError("a backtest target lies after the origin")
                sec = float(pd.read_parquet(bt["timing"])["seconds"].iloc[0])
                log(f"{name}: backtests written ({N_INNER} calls, {sec:.0f}s of model time)")
            status["jobs"][name] = f"done in {time.perf_counter() - t0:.0f}s"
        except Exception as e:                                                       # noqa: BLE001
            failed += 1
            status["jobs"][name] = f"FAILED: {type(e).__name__}: {str(e)[-300:]}"
            log(f"{name}: FAILED {type(e).__name__}: {str(e)[-300:]}")
            traceback.print_exc()
        write_status(status)

    status["state"] = "finished" if failed == 0 else f"finished with {failed} failures"
    write_status(status)
    log(f"FINISHED: {len(jobs) - failed} of {len(jobs)} jobs done, {failed} failed")


if __name__ == "__main__":
    main()
