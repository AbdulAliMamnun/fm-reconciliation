"""Dataset loading in the format `hierarchicalforecast` expects.

This module must not import hierarchicalforecast: it is used by the scripts
that run foundation models (see src/runners.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from datasetsforecast.hierarchical import HierarchicalData

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# freq, horizon (PREREG §3) and season length m per dataset.
DATASETS = {
    "TourismSmall": {"freq": "QE", "h": 8, "m": 4},
    "TourismLarge": {"freq": "MS", "h": 12, "m": 12},
    "Labour": {"freq": "MS", "h": 12, "m": 12},
    "M5": {"freq": "D", "h": 28, "m": 7},
}

# PREREG §3: "M5 subset (1-2 stores, item -> dept -> category -> store)". The
# registration does not name the stores. These are the first in the order of the
# M5 files.
M5_STORES = ("CA_1",)
M5_LEVELS = ["Store", "Store/Category", "Store/Category/Department", "Store/Category/Department/Item"]


def load_m5(stores=M5_STORES):
    """The M5 subset as a hierarchy: item -> department -> category -> store.

    Days d_1 to d_1969 (the evaluation training file and the evaluation test
    file), the same days as datasetsforecast.m5. Unlike that loader, leading
    zeros are kept: every series has all 1,969 days, so that the items add up
    to their department on every day. With more than one store there is no
    level above the stores.

    Returns (Y_df, S_df, tags) in the format of `load_hierarchy`.
    """
    from datasetsforecast.m5 import M5

    stores = tuple(stores)
    cache = DATA_DIR / "m5" / f"subset_{'_'.join(stores)}.parquet"
    if cache.exists():
        sales = pd.read_parquet(cache)
    else:
        M5.download(str(DATA_DIR))
        path = DATA_DIR / "m5" / "datasets"
        keys = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
        train = pd.read_csv(path / "sales_train_evaluation.csv").drop(columns="id", errors="ignore")
        test = pd.read_csv(path / "sales_test_evaluation.csv").drop(columns="id", errors="ignore")
        sales = train.merge(test, on=keys, how="left", validate="one_to_one")
        unknown = set(stores) - set(sales["store_id"])
        if unknown:
            raise ValueError(f"Unknown M5 stores {sorted(unknown)}.")
        sales = pd.concat([sales[sales["store_id"] == st] for st in stores], ignore_index=True)
        sales.to_parquet(cache, index=False)

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    # Row i of the calendar is day d_{i+1}; this copy of the file has no `d` column.
    cal = pd.read_csv(DATA_DIR / "m5" / "datasets" / "calendar.csv", usecols=["date"],
                      parse_dates=["date"])["date"]
    dates = pd.DatetimeIndex(cal.iloc[[int(c[2:]) - 1 for c in day_cols]])
    values = sales[day_cols].to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError("The M5 subset has missing values.")

    names = [
        sales["store_id"],
        sales["store_id"] + "/" + sales["cat_id"],
        sales["store_id"] + "/" + sales["cat_id"] + "/" + sales["dept_id"],
        sales["store_id"] + "/" + sales["cat_id"] + "/" + sales["dept_id"] + "/" + sales["item_id"],
    ]
    bottom = names[-1].to_numpy()
    if len(set(bottom)) != len(bottom):
        raise ValueError("Item ids are not unique within a store.")
    tags, rows, ids = {}, [], []
    for level, name in zip(M5_LEVELS, names):
        members = list(dict.fromkeys(name))
        tags[level] = np.array(members, dtype=object)
        onehot = (name.to_numpy()[None, :] == np.array(members, dtype=object)[:, None])
        rows.append(onehot.astype(np.float64))
        ids += members
    S = np.vstack(rows)
    S_df = pd.DataFrame(S, columns=bottom)
    S_df.insert(0, "unique_id", ids)

    Y = S @ values                                        # (n_series, n_days)
    Y_df = pd.DataFrame({
        "unique_id": np.repeat(np.asarray(ids, dtype=object), len(dates)),
        "ds": np.tile(dates, len(ids)),
        "y": Y.reshape(-1),
    })
    return Y_df, S_df, tags


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

    Labour ends at 2019-12: the datasetsforecast loader drops 2020 onwards.
    M5 is the subset of `load_m5`.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Known: {sorted(DATASETS)}")
    cfg = DATASETS[name]
    if name == "M5":
        Y_df, S_df, tags = load_m5()
        return Y_df, S_df, tags, cfg["freq"], cfg["h"]

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

    return Y_df, S_df, tags, cfg["freq"], cfg["h"]
