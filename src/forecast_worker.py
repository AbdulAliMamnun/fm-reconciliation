"""Run a model in its own conda env and write the forecasts to parquet.

Used for models whose dependencies conflict with the main env. Called by
src/forecasters.py as

    conda run -n <env> python -m src.forecast_worker --model ... --input ... --output ...

from the repo root. It imports only src.runners.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.runners import run_cutoffs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True, help="parquet with unique_id, ds, y")
    p.add_argument("--output", required=True, help="parquet to write the forecasts to")
    p.add_argument("--info", required=True, help="json to write the timing to")
    p.add_argument("--cutoffs", required=True, help="comma-separated dates")
    p.add_argument("--h", type=int, required=True)
    p.add_argument("--freq", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--samples-dir", default=None)
    p.add_argument("--trim-leading-zeros", action="store_true")
    a = p.parse_args()

    y_df = pd.read_parquet(a.input)
    cutoffs = [pd.Timestamp(c) for c in a.cutoffs.split(",")]
    forecasts, info = run_cutoffs(a.model, y_df, cutoffs, a.h, a.freq, device=a.device,
                                  keep_samples=a.samples_dir is not None,
                                  trim_leading_zeros_=a.trim_leading_zeros)
    forecasts.to_parquet(a.output, index=False)
    for cutoff, frame in info.pop("samples").items():
        frame.to_parquet(Path(a.samples_dir) / f"{cutoff.date()}.parquet", index=False)
    Path(a.info).write_text(json.dumps(info))


if __name__ == "__main__":
    main()
