"""Dataset loading in the format `hierarchicalforecast` expects."""
from pathlib import Path

import pandas as pd
from datasetsforecast.hierarchical import HierarchicalData

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# freq and horizon per dataset (PREREG §3).
DATASETS = {
    "TourismSmall": {"freq": "QE", "h": 8},
    "TourismLarge": {"freq": "MS", "h": 12},
}


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
