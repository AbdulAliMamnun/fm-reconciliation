"""Hand-computed examples for src/metrics.py."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import level_mean, mase, rmsse, scale_mase, scale_rmsse, scales_after_first_nonzero


def test_scale_mase_hand_computed():
    # m=2: diffs are 3-1, 2-4, 7-3, 1-2 = 2, -2, 4, -1 -> mean |.| = 9/4
    y_train = [1, 4, 3, 2, 7, 1]
    assert scale_mase(y_train, 2) == pytest.approx(2.25)


def test_scale_rmsse_hand_computed():
    # same diffs squared: 4, 4, 16, 1 -> mean = 25/4
    y_train = [1, 4, 3, 2, 7, 1]
    assert scale_rmsse(y_train, 2) == pytest.approx(6.25)


def test_scale_m1_is_mean_abs_first_difference():
    # diffs: 2, -3, 5 -> mean |.| = 10/3, mean sq = 38/3
    y_train = [1, 3, 0, 5]
    assert scale_mase(y_train, 1) == pytest.approx(10 / 3)
    assert scale_rmsse(y_train, 1) == pytest.approx(38 / 3)


def test_scale_uses_only_the_data_passed():
    # Appending a holdout changes the scale, so the caller must pass training only.
    train = [1, 4, 3, 2, 7, 1]
    assert scale_mase(train + [100, 100], 2) != pytest.approx(scale_mase(train, 2))


def test_scale_is_zero_for_a_perfectly_seasonal_series():
    y_train = [5, 0, 5, 0, 5, 0]
    assert scale_mase(y_train, 2) == 0.0
    assert scale_rmsse(y_train, 2) == 0.0


def test_scale_needs_more_than_m_observations():
    with pytest.raises(ValueError):
        scale_mase([1, 2], 2)


def test_scale_2d_is_per_series():
    Y = np.array([[1, 4, 3, 2, 7, 1], [0, 0, 1, 1, 2, 2]])
    # second row, m=2: diffs 1, 1, 1, 1
    np.testing.assert_allclose(scale_mase(Y, 2), [2.25, 1.0])
    np.testing.assert_allclose(scale_rmsse(Y, 2), [6.25, 1.0])


def test_mase_hand_computed():
    # |errors| = 1, 2, 3 -> mean 2; scale 4 -> 0.5
    assert mase([10, 10, 10], [9, 12, 7], 4.0) == pytest.approx(0.5)


def test_rmsse_hand_computed():
    # squared errors = 1, 4, 9 -> mean 14/3; scale_sq 6 -> sqrt(7/9)
    assert rmsse([10, 10, 10], [9, 12, 7], 6.0) == pytest.approx(np.sqrt(7 / 9))


def test_perfect_forecast_scores_zero():
    assert mase([1, 2, 3], [1, 2, 3], 2.0) == 0.0
    assert rmsse([1, 2, 3], [1, 2, 3], 2.0) == 0.0


def test_seasonal_naive_in_sample_scores_one():
    # Scoring the seasonal-naive forecast on the data that defined the scale gives 1.
    y = np.array([1.0, 4, 3, 2, 7, 1])
    m = 2
    assert mase(y[m:], y[:-m], scale_mase(y, m)) == pytest.approx(1.0)
    assert rmsse(y[m:], y[:-m], scale_rmsse(y, m)) == pytest.approx(1.0)


def test_mase_2d_is_per_series():
    y = np.array([[10, 10], [0, 0]])
    yhat = np.array([[8, 14], [1, 1]])
    # row 0: mean |.| = 3, scale 2 -> 1.5 ; row 1: mean |.| = 1, scale 4 -> 0.25
    np.testing.assert_allclose(mase(y, yhat, np.array([2.0, 4.0])), [1.5, 0.25])
    # row 0: mean sq = 10, scale_sq 5 -> sqrt(2) ; row 1: mean sq = 1, scale_sq 4 -> 0.5
    np.testing.assert_allclose(rmsse(y, yhat, np.array([5.0, 4.0])), [np.sqrt(2), 0.5])


TAGS = {
    "Top": np.array(["T"], dtype=object),
    "Bottom": np.array(["a", "b", "c"], dtype=object),
}


def test_level_mean_hand_computed():
    df = pd.DataFrame(
        {
            "unique_id": ["T", "a", "b", "c"],
            "scale": [10.0, 1.0, 2.0, 3.0],
            "mase": [0.5, 1.0, 2.0, 6.0],
        }
    )
    out = level_mean(df, TAGS)
    assert list(out["level"]) == ["Top", "Bottom"]
    assert list(out["n_series"]) == [1, 3]
    assert list(out["n_excluded"]) == [0, 0]
    assert out["mase"].tolist() == pytest.approx([0.5, 3.0])


def test_level_mean_excludes_and_counts_zero_scale_series():
    df = pd.DataFrame(
        {
            "unique_id": ["T", "a", "b", "c"],
            "scale": [10.0, 1.0, 0.0, 5e-9],
            "mase": [0.5, 1.0, np.inf, 1e9],
            "rmsse": [0.4, 2.0, np.inf, 1e9],
        }
    )
    out = level_mean(df, TAGS).set_index("level")
    assert out.loc["Bottom", "n_series"] == 1
    assert out.loc["Bottom", "n_excluded"] == 2
    assert out.loc["Bottom", "mase"] == pytest.approx(1.0)
    assert out.loc["Bottom", "rmsse"] == pytest.approx(2.0)
    assert out.loc["Top", "n_excluded"] == 0


def test_level_mean_threshold_is_strictly_less_than_tol():
    df = pd.DataFrame(
        {"unique_id": ["T", "a", "b", "c"], "scale": [1.0, 1e-8, 1.0, 1.0], "mase": [1.0, 3.0, 0.0, 0.0]}
    )
    out = level_mean(df, TAGS).set_index("level")
    assert out.loc["Bottom", "n_excluded"] == 0
    assert out.loc["Bottom", "mase"] == pytest.approx(1.0)


def test_level_mean_all_excluded_gives_nan():
    df = pd.DataFrame(
        {"unique_id": ["T", "a", "b", "c"], "scale": [0.0, 1.0, 1.0, 1.0], "mase": [np.inf, 1.0, 1.0, 1.0]}
    )
    out = level_mean(df, TAGS).set_index("level")
    assert out.loc["Top", "n_series"] == 0
    assert out.loc["Top", "n_excluded"] == 1
    assert np.isnan(out.loc["Top", "mase"])


def test_level_mean_grouped_by_model():
    df = pd.DataFrame(
        {
            "model": ["A"] * 4 + ["B"] * 4,
            "unique_id": ["T", "a", "b", "c"] * 2,
            "scale": [10.0, 1.0, 2.0, 3.0] * 2,
            "mase": [0.5, 1.0, 2.0, 6.0, 1.5, 2.0, 2.0, 2.0],
        }
    )
    out = level_mean(df, TAGS, by=["model"]).set_index(["model", "level"])
    assert out.loc[("A", "Bottom"), "mase"] == pytest.approx(3.0)
    assert out.loc[("B", "Bottom"), "mase"] == pytest.approx(2.0)
    assert out.loc[("B", "Top"), "mase"] == pytest.approx(1.5)


def test_level_mean_raises_on_missing_series():
    df = pd.DataFrame({"unique_id": ["T", "a", "b"], "scale": [1.0, 1.0, 1.0], "mase": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="missing"):
        level_mean(df, TAGS)


def test_level_mean_raises_on_duplicate_rows():
    df = pd.DataFrame(
        {"unique_id": ["T", "a", "b", "c", "c"], "scale": [1.0] * 5, "mase": [1.0] * 5}
    )
    with pytest.raises(ValueError, match="duplicate"):
        level_mean(df, TAGS)


def test_scales_after_first_nonzero_hand_computed():
    Y = np.array([
        [0.0, 0.0, 1.0, 4.0, 3.0, 2.0, 7.0, 1.0],    # starts at index 2; then the series of the first test
        [1.0, 4.0, 3.0, 2.0, 7.0, 1.0, 0.0, 0.0],    # starts at 0; zeros at the end stay
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # never sells
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 2.0],    # 2 values after the first sale, m = 2 needs 3
    ])
    mase_s, rmsse_s, start = scales_after_first_nonzero(Y, 2)
    assert mase_s[0] == pytest.approx(2.25) and rmsse_s[0] == pytest.approx(6.25)
    assert mase_s[1] == pytest.approx(scale_mase(Y[1], 2)) and rmsse_s[1] == pytest.approx(scale_rmsse(Y[1], 2))
    assert np.isnan(mase_s[2]) and np.isnan(rmsse_s[2])
    assert np.isnan(mase_s[3]) and np.isnan(rmsse_s[3])
    np.testing.assert_array_equal(start, [2, 0, 8, 6])


def test_level_mean_excludes_series_without_a_scale():
    df = pd.DataFrame({"unique_id": ["T", "a", "b", "c"], "scale": [1.0, 1.0, np.nan, 2.0],
                       "mase": [1.0, 2.0, 99.0, 4.0]})
    out = level_mean(df, {"Top": np.array(["T"], dtype=object), "Bottom": np.array(["a", "b", "c"], dtype=object)})
    out = out.set_index("level")
    assert out.loc["Bottom", "n_excluded"] == 1 and out.loc["Bottom", "n_series"] == 2
    assert out.loc["Bottom", "mase"] == pytest.approx(3.0)


def test_unsold_items_are_excluded_from_metrics_and_counted():
    # scales_after_first_nonzero gives NaN to an item with no sale; level_mean drops and counts it
    Y = np.array([[5.0, 6, 7, 8, 9, 10], [1.0, 2, 1, 2, 1, 2], [0.0, 0, 0, 0, 0, 0]])
    mase_s, rmsse_s, start = scales_after_first_nonzero(Y, 1)
    unsold = start == Y.shape[1]
    np.testing.assert_array_equal(unsold, [False, False, True])
    df = pd.DataFrame({"unique_id": ["T", "a", "b"], "scale": mase_s, "mase": [1.0, 2.0, 0.0],
                       "rmsse": [1.0, 3.0, 0.0]})
    out = level_mean(df, {"Top": np.array(["T"], dtype=object),
                          "Bottom": np.array(["a", "b"], dtype=object)}).set_index("level")
    assert out.loc["Bottom", "n_excluded"] == 1 and out.loc["Bottom", "n_series"] == 1
    assert out.loc["Bottom", "mase"] == pytest.approx(2.0) and out.loc["Bottom", "rmsse"] == pytest.approx(3.0)
