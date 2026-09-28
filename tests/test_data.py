import numpy as np
import pandas as pd
import pytest

from src.data import M5_LEVELS, load_hierarchy, load_m5, rolling_origins

EXPECTED = {
    "TourismSmall": {"freq": "QE", "h": 8, "n_series": 89, "n_bottom": 56},
    "TourismLarge": {"freq": "MS", "h": 12, "n_series": 555, "n_bottom": 304},
    "Labour": {"freq": "MS", "h": 12, "n_series": 57, "n_bottom": 32},
    "M5": {"freq": "D", "h": 28, "n_series": 3060, "n_bottom": 3049},
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


def test_rolling_origins_tourism_large():
    Y_df, _, _, _, h = load_hierarchy("TourismLarge")
    splits = rolling_origins(Y_df, h, 5)
    assert [str(s["origin"].date()) for s in splits] == [
        "2011-12-01", "2012-12-01", "2013-12-01", "2014-12-01", "2015-12-01"]
    for i, s in enumerate(splits):
        assert s["train"]["ds"].max() == s["origin"]
        assert s["train"]["ds"].nunique() == 168 + 12 * i
        assert s["test"]["ds"].nunique() == h
        assert s["test"]["ds"].min() > s["origin"]
        assert s["train"]["unique_id"].nunique() == 555
        assert len(s["test"]) == 555 * h
    # origins are h apart, test windows do not overlap, and the last one ends at the last observation
    assert splits[-1]["test"]["ds"].max() == Y_df["ds"].max()
    for a, b in zip(splits[:-1], splits[1:]):
        assert a["test"]["ds"].max() == b["origin"]


def test_rolling_origins_raises_when_too_short():
    Y_df, _, _, _, h = load_hierarchy("TourismSmall")
    with pytest.raises(ValueError):
        rolling_origins(Y_df, h, 5)


def test_labour_range_and_levels():
    Y_df, _, tags, _, _ = load_hierarchy("Labour")
    assert str(Y_df["ds"].min().date()) == "1978-02-01"
    assert str(Y_df["ds"].max().date()) == "2019-12-01"          # the loader drops 2020 onwards
    assert [len(v) for v in tags.values()] == [1, 8, 16, 32]


def test_m5_subset_is_one_store_with_four_levels():
    Y_df, S_df, tags, _, _ = load_hierarchy("M5")
    assert list(tags) == M5_LEVELS
    assert [len(v) for v in tags.values()] == [1, 3, 7, 3049]
    assert list(tags["Store"]) == ["CA_1"]
    assert list(tags["Store/Category"]) == ["CA_1/HOBBIES", "CA_1/HOUSEHOLD", "CA_1/FOODS"]
    assert Y_df["ds"].nunique() == 1969
    assert str(Y_df["ds"].min().date()) == "2011-01-29"
    assert str(Y_df["ds"].max().date()) == "2016-06-19"
    assert Y_df.groupby("unique_id").size().eq(1969).all()       # leading zeros are kept
    # a department is the sum of its items, a category of its departments
    bottom = [c for c in S_df.columns if c != "unique_id"]
    S = S_df.set_index("unique_id")
    dept = S.loc["CA_1/FOODS/FOODS_1", bottom]
    assert dept.sum() == sum(b.startswith("CA_1/FOODS/FOODS_1/") for b in bottom)
    assert set(dept[dept == 1].index) == {b for b in bottom if b.startswith("CA_1/FOODS/FOODS_1/")}
    assert S.loc["CA_1", bottom].sum() == 3049


def test_m5_two_stores_has_no_level_above_the_stores():
    Y_df, S_df, tags = load_m5(("CA_1", "CA_2"))
    assert [len(v) for v in tags.values()] == [2, 6, 14, 6098]
    assert S_df.shape == (6120, 6099)
    assert list(tags["Store"]) == ["CA_1", "CA_2"]
    S = S_df.set_index("unique_id")
    assert S.loc["CA_1"].sum() == 3049 and S.loc["CA_2"].sum() == 3049
    assert (S.loc["CA_1"] * S.loc["CA_2"]).sum() == 0
