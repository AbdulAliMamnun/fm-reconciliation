"""Tests for the long results table and the metrics computed from it."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import level_mean, mase, rmsse, series_metrics
from src.results import COLUMNS, QUANTILE_COLS, append_results, build_results

TAGS = {
    "Top": np.array(["T"], dtype=object),
    "Bottom": np.array(["a", "b"], dtype=object),
}
ORIGIN = pd.Timestamp("2015-12-01")
DATES = pd.to_datetime(["2016-01-01", "2016-02-01"])

# y: T = a + b
Y = {"T": [10.0, 20.0], "a": [4.0, 8.0], "b": [6.0, 12.0]}
BASE = {"T": [12.0, 17.0], "a": [5.0, 8.0], "b": [6.0, 10.0]}
BU = {"T": [11.0, 18.0], "a": [5.0, 8.0], "b": [6.0, 10.0]}
SCALES = pd.DataFrame(
    {"unique_id": ["T", "a", "b"], "mase_scale": [2.0, 1.0, 4.0], "rmsse_scale": [8.0, 2.0, 16.0]}
)


def _long(values, name):
    return pd.DataFrame(
        [(uid, d, v) for uid, vals in values.items() for d, v in zip(DATES, vals)],
        columns=["unique_id", "ds", name],
    )


def _forecasts():
    # Deliberately shuffled: build_results must not depend on row order.
    f = _long(BASE, "M").merge(_long(BU, "M/BottomUp"), on=["unique_id", "ds"])
    return f.sample(frac=1, random_state=0).reset_index(drop=True)


def _build(origin=ORIGIN, forecasts=None, point="median"):
    return build_results(
        dataset="Toy", origin=origin, model="M",
        forecasts=_forecasts() if forecasts is None else forecasts,
        actuals=_long(Y, "y"), tags=TAGS, scales=SCALES,
        w_est={"M": "base", "M/BottomUp": "bottomup"}, point=point,
    )


def test_build_results_schema_and_shape():
    res = _build()
    assert list(res.columns) == COLUMNS
    assert len(res) == 2 * 3 * 2  # W_est x series x horizon
    assert res[QUANTILE_COLS].isna().all().all()
    assert set(res["W_est"]) == {"base", "bottomup"}
    assert (res["origin"] == ORIGIN).all()
    assert (res["point"] == "median").all()


def test_build_results_values_are_aligned():
    res = _build().set_index(["W_est", "series_id", "horizon"])
    assert res.loc[("base", "T", 1), "yhat"] == 12.0
    assert res.loc[("base", "T", 2), "yhat"] == 17.0
    assert res.loc[("bottomup", "T", 2), "yhat"] == 18.0
    assert res.loc[("base", "b", 2), "y"] == 12.0
    assert res.loc[("base", "b", 2), "yhat"] == 10.0
    assert res.loc[("base", "a", 1), "level"] == "Bottom"
    assert res.loc[("base", "T", 1), "level"] == "Top"
    assert res.loc[("bottomup", "b", 1), "mase_scale"] == 4.0
    assert res.loc[("bottomup", "b", 1), "rmsse_scale"] == 16.0


def test_build_results_raises_without_actuals():
    f = _forecasts()
    f["ds"] = f["ds"] + pd.DateOffset(years=5)
    with pytest.raises(ValueError, match="actual"):
        _build(forecasts=f)


def test_build_results_raises_if_forecast_not_after_origin():
    with pytest.raises(ValueError, match="origin"):
        _build(origin=pd.Timestamp("2016-01-01"))


def test_append_results_is_idempotent_and_keeps_other_keys(tmp_path):
    path = tmp_path / "results.parquet"
    first = _build()
    other = _build(origin=pd.Timestamp("2015-11-01"))
    append_results(path, first)
    append_results(path, other)
    full = append_results(path, first)  # same keys again: replaced, not duplicated
    assert len(full) == len(first) + len(other)
    assert not full.duplicated(["dataset", "origin", "model", "W_est", "point", "series_id", "horizon"]).any()
    assert set(full["origin"]) == {ORIGIN, pd.Timestamp("2015-11-01")}
    pd.testing.assert_frame_equal(pd.read_parquet(path), full)


def test_append_results_replaces_values_for_the_same_key(tmp_path):
    path = tmp_path / "results.parquet"
    append_results(path, _build())
    changed = _build()
    changed["yhat"] = 0.0
    full = append_results(path, changed)
    assert len(full) == len(changed)
    assert (full["yhat"] == 0.0).all()


def test_append_results_rejects_wrong_columns(tmp_path):
    with pytest.raises(ValueError):
        append_results(tmp_path / "r.parquet", _build().drop(columns="q10"))


def test_series_metrics_hand_computed():
    sm = series_metrics(_build()).set_index(["W_est", "series_id"])
    # base, T: errors (yhat - y) = +2, -3 -> mae 2.5, mse 6.5, bias -0.5
    assert sm.loc[("base", "T"), "mase"] == pytest.approx(2.5 / 2.0)
    assert sm.loc[("base", "T"), "rmsse"] == pytest.approx(np.sqrt(6.5 / 8.0))
    assert sm.loc[("base", "T"), "bias"] == pytest.approx(-0.5)
    # bottomup, T: errors +1, -2 -> mae 1.5, mse 2.5, bias -0.5
    assert sm.loc[("bottomup", "T"), "mase"] == pytest.approx(0.75)
    assert sm.loc[("bottomup", "T"), "rmsse"] == pytest.approx(np.sqrt(2.5 / 8.0))
    # base, b: errors 0, -2 -> mae 1, mse 2, bias -1
    assert sm.loc[("base", "b"), "mase"] == pytest.approx(0.25)
    assert sm.loc[("base", "b"), "rmsse"] == pytest.approx(np.sqrt(2 / 16))
    assert sm.loc[("base", "b"), "bias"] == pytest.approx(-1.0)
    assert (sm["n_horizons"] == 2).all()


def test_series_metrics_agree_with_array_functions():
    sm = series_metrics(_build()).set_index(["W_est", "series_id"])
    scales = SCALES.set_index("unique_id")
    for uid in Y:
        assert sm.loc[("base", uid), "mase"] == pytest.approx(
            mase(Y[uid], BASE[uid], scales.loc[uid, "mase_scale"]))
        assert sm.loc[("base", uid), "rmsse"] == pytest.approx(
            rmsse(Y[uid], BASE[uid], scales.loc[uid, "rmsse_scale"]))


def test_level_mean_from_the_results_table():
    sm = series_metrics(_build())
    tags = {**TAGS, "All series": np.array(["T", "a", "b"], dtype=object)}
    lm = level_mean(sm, tags, value_cols=["mase"], by=["W_est"], scale_col="mase_scale",
                    id_col="series_id").set_index(["W_est", "level"])
    # base: T 1.25, a (errors 1, 0 -> mae .5 / 1) 0.5, b 0.25
    assert lm.loc[("base", "Top"), "mase"] == pytest.approx(1.25)
    assert lm.loc[("base", "Bottom"), "mase"] == pytest.approx(0.375)
    assert lm.loc[("base", "All series"), "mase"] == pytest.approx((1.25 + 0.5 + 0.25) / 3)
    assert lm.loc[("base", "All series"), "n_series"] == 3


def test_build_results_with_quantiles_for_base_only():
    q = _long(BASE, "q50").assign(W_est="base")
    for i, col in enumerate(QUANTILE_COLS):
        q[col] = q["q50"] + (i - 4)
    q = q.sample(frac=1, random_state=1)
    res = build_results(
        dataset="Toy", origin=ORIGIN, model="M", forecasts=_forecasts(), actuals=_long(Y, "y"),
        tags=TAGS, scales=SCALES, w_est={"M": "base", "M/BottomUp": "bottomup"},
        point="median", quantiles=q,
    )
    assert list(res.columns) == COLUMNS
    assert len(res) == 12
    base = res[res.W_est == "base"].set_index(["series_id", "horizon"])
    assert base.loc[("T", 2), "q50"] == 17.0
    assert base.loc[("T", 2), "q10"] == 13.0
    assert base.loc[("b", 1), "q90"] == 10.0
    assert (base["q50"] == base["yhat"]).all()
    assert res.loc[res.W_est == "bottomup", QUANTILE_COLS].isna().all().all()
    assert res[QUANTILE_COLS].dtypes.eq("float64").all()


def test_build_results_rejects_unknown_quantile_label():
    q = _long(BASE, "q50").assign(W_est="something_else")
    for col in QUANTILE_COLS:
        q[col] = 1.0
    with pytest.raises(ValueError):
        build_results(
            dataset="Toy", origin=ORIGIN, model="M", forecasts=_forecasts(), actuals=_long(Y, "y"),
            tags=TAGS, scales=SCALES, w_est={"M": "base", "M/BottomUp": "bottomup"},
            point="median", quantiles=q,
        )


def test_build_results_rejects_unknown_point():
    with pytest.raises(ValueError, match="point"):
        _build(point="mode")


def test_median_and_mean_arms_are_separate_keys(tmp_path):
    path = tmp_path / "results.parquet"
    median = _build(point="median")
    mean = _build(point="mean")
    mean["yhat"] = mean["yhat"] + 1.0
    append_results(path, median)
    full = append_results(path, mean)
    assert len(full) == len(median) + len(mean)
    # re-appending one arm replaces only that arm
    full = append_results(path, mean)
    assert len(full) == len(median) + len(mean)
    assert (full.loc[full.point == "median", "yhat"].to_numpy() == median["yhat"].to_numpy()).all()
    sm = series_metrics(full).set_index(["point", "W_est", "series_id"])
    assert sm.loc[("median", "base", "T"), "bias"] == pytest.approx(-0.5)
    assert sm.loc[("mean", "base", "T"), "bias"] == pytest.approx(0.5)


def test_append_results_refuses_an_old_schema(tmp_path):
    path = tmp_path / "results.parquet"
    _build().drop(columns="point").to_parquet(path, index=False)
    with pytest.raises(ValueError, match="older schema"):
        append_results(path, _build())


def test_mean9_is_a_third_arm(tmp_path):
    path = tmp_path / "results.parquet"
    for point in ["median", "mean", "mean9"]:
        full = append_results(path, _build(point=point))
    assert len(full) == 3 * len(_build())
    assert set(full["point"]) == {"median", "mean", "mean9"}
    assert len(series_metrics(full)) == 3 * 2 * 3           # arms x W_est x series
