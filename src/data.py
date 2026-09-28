"""Dataset loading in the format `hierarchicalforecast` expects."""
from pathlib import Path

import pandas as pd
from datasetsforecast.hierarchical import HierarchicalData

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# freq, horizon (PREREG §3) and season length m per dataset.
DATASETS = {
    "TourismSmall": {"freq": "QE", "h": 8, "m": 4},
    "TourismLarge": {"freq": "MS", "h": 12, "m": 12},
}


def rolling_origins(Y_df, h, n_origins):
    """Split Y_df at rolling origins spaced h steps apart, ending at the last
    observation (PREREG §3). Earliest origin first.

    Returns a list of dicts with `origin` (the last training timestamp),
    `train` (all rows up to and including the origin) and `test` (the next h
    time points).
    """
    dates = pd.DatetimeIndex(sorted(Y_df["ds"].unique()))
    T = len(dates)
    if T - h * n_origins < 1:
        raise ValueError(f"{T} time points are too few for {n_origins} origins with h={h}.")
    out = []
    for k in range(n_origins, 0, -1):
        n_train = T - h * k
        origin = dates[n_train - 1]
        test_dates = dates[n_train : n_train + h]
        out.append({
            "origin": origin,
            "train": Y_df[Y_df["ds"] <= origin],
            "test": Y_df[Y_df["ds"].isin(test_dates)],
        })
    return out


def load_hierarchy(name):
    """Load a hierarchical dataset.

    Returns (Y_df, S_df, tags, freq, h).

    Y_df: long frame with columns unique_id, ds (datetime), y, holding every
        series at every level, sorted in S_df row order and then by ds.
    S_df: summing matrix with a `unique_id` column followed by one column per
        bottom series. Rows run from the total down to the bottom level, so
        the last n_bottom rows are an identity.
    tags: dict mapping level name -> array of the unique_ids at that level.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Known: {sorted(DATASETS)}")

    Y_df, S_df, tags = HierarchicalData.load(str(DATA_DIR), name)

    S_df = S_df.reset_index(names="unique_id")

    Y_df = Y_df.copy()
    Y_df["ds"] = pd.to_datetime(Y_df["ds"])
    Y_df["y"] = Y_df["y"].astype("float64")
    order = pd.Categorical(Y_df["unique_id"], categories=S_df["unique_id"], ordered=True)
    Y_df = (
        Y_df.assign(_order=order.codes)
        .sort_values(["_order", "ds"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    cfg = DATASETS[name]
    return Y_df, S_df, tags, cfg["freq"], cfg["h"]
