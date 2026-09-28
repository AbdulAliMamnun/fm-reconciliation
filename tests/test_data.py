import numpy as np
import pandas as pd
import pytest

from src.data import load_hierarchy

EXPECTED = {
    "TourismSmall": {"freq": "QE", "h": 8, "n_series": 89, "n_bottom": 56},
    "TourismLarge": {"freq": "MS", "h": 12, "n_series": 555, "n_bottom": 304},
}


@pytest.fixture(scope="module", params=list(EXPECTED))
def loaded(request):
    return request.param, load_hierarchy(request.param)


def test_freq_and_horizon(loaded):
    name, (_, _, _, freq, h) = loaded
    assert freq == EXPECTED[name]["freq"]
    assert h == EXPECTED[name]["h"]


def test_S_df_has_unique_id(loaded):
    name, (_, S_df, _, _, _) = loaded
    assert S_df.columns[0] == "unique_id"
    assert S_df["unique_id"].is_unique
    assert S_df.shape == (EXPECTED[name]["n_series"], EXPECTED[name]["n_bottom"] + 1)


def test_bottom_block_is_identity(loaded):
    _, (_, S_df, _, _, _) = loaded
    bottom_ids = [c for c in S_df.columns if c != "unique_id"]
    n = len(bottom_ids)
    S = S_df[bottom_ids].to_numpy()
    assert np.array_equal(S[-n:], np.eye(n))
    assert list(S_df["unique_id"].iloc[-n:]) == bottom_ids


def test_S_times_bottom_reproduces_Y(loaded):
    _, (Y_df, S_df, _, _, _) = loaded
    bottom_ids = [c for c in S_df.columns if c != "unique_id"]
    S = S_df[bottom_ids].to_numpy()
    wide = Y_df.pivot(index="unique_id", columns="ds", values="y")
    Y = wide.loc[S_df["unique_id"]].to_numpy()
    B = wide.loc[bottom_ids].to_numpy()
    assert not np.isnan(Y).any()
    np.testing.assert_allclose(S @ B, Y, rtol=0, atol=1e-8)


def test_tags_concatenated_equal_S_row_order(loaded):
    _, (_, S_df, tags, _, _) = loaded
    assert list(np.concatenate(list(tags.values()))) == list(S_df["unique_id"])


def test_ds_is_datetime_and_matches_freq(loaded):
    _, (Y_df, _, _, freq, _) = loaded
    assert pd.api.types.is_datetime64_any_dtype(Y_df["ds"])
    dates = pd.DatetimeIndex(np.sort(Y_df["ds"].unique()))
    expected = pd.date_range(dates[0], periods=len(dates), freq=freq)
    assert dates.equals(expected)


def test_Y_df_order_matches_S_df(loaded):
    _, (Y_df, S_df, _, _, _) = loaded
    assert list(Y_df["unique_id"].unique()) == list(S_df["unique_id"])
    assert Y_df.groupby("unique_id", sort=False)["ds"].apply(lambda s: s.is_monotonic_increasing).all()


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_hierarchy("NotADataset")
