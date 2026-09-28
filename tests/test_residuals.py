import numpy as np
import pandas as pd
import pytest

from src import forecasters
from src.forecasters import QUANTILE_LEVELS
from src.residuals import (
    RESIDUAL_COLUMNS, backtest_paths, backtest_residuals, inner_forecasts, inner_origins,
    residual_array,
)

H, N_INNER, T = 12, 24, 80
DATES = pd.date_range("2010-01-01", periods=T, freq="MS")
OUTER = DATES[-1]
IDS = ["b", "a", "c"]


def _y(extra=0):
    """y_t = t + 1000 * (series number), so a value identifies its time and series.
    `extra` appends periods after the outer origin, as a test window would."""
    ds = pd.date_range("2010-01-01", periods=T + extra, freq="MS")
    return pd.concat(
        [pd.DataFrame({"unique_id": uid, "ds": ds, "y": np.arange(len(ds), dtype=float) + 1000 * k})
         for k, uid in enumerate(IDS)], ignore_index=True)


@pytest.fixture
def last_value_model(monkeypatch):
    """Forecasts the last value it was given, with quantiles q = last + level - 0.5.
    Records the last context value of every call."""
    seen = []

    def run(contexts, h, device):
        last = np.array([c[-1] for c in contexts])
        seen.append((len(contexts[0]), last.copy()))
        return last[:, None, None] + np.zeros((1, h, 1)) + (np.array(QUANTILE_LEVELS) - 0.5)

    monkeypatch.setitem(forecasters.MODELS, "last_value", (run, "none/last_value"))
    return seen


def test_inner_origins_are_the_registered_ones():
    inner = inner_origins(DATES, H, N_INNER)
    assert len(inner) == N_INNER
    assert inner[-1] == DATES[-1 - H]                   # t_o - h
    assert inner[0] == DATES[-1 - H - (N_INNER - 1)]    # t_o - h - 23
    assert (np.diff(inner.to_numpy()) > np.timedelta64(0)).all()
    assert list(inner) == list(DATES[T - 1 - H - 23: T - H])


def test_inner_origins_raise_when_history_is_too_short():
    with pytest.raises(ValueError):
        inner_origins(DATES[:35], H, N_INNER)   # needs at least 1 + 23 + 12 = 36 periods
    assert len(inner_origins(DATES[:37], H, N_INNER)) == N_INNER


def test_no_residual_target_exceeds_the_outer_origin(last_value_model):
    r = backtest_residuals("last_value", _y(), H, N_INNER)
    assert list(r.columns) == RESIDUAL_COLUMNS
    assert r["ds"].max() == OUTER                       # the last target is exactly t_o
    assert (r["ds"] <= OUTER).all()
    assert (r["ds"] > r["inner_origin"]).all()
    assert (r["inner_origin"] <= OUTER - pd.DateOffset(months=H)).all()
    assert len(r) == len(IDS) * N_INNER * H
    assert r.groupby(["unique_id", "horizon"]).size().eq(N_INNER).all()
    # target date = inner origin + horizon
    months = (r["ds"].dt.year - r["inner_origin"].dt.year) * 12 + r["ds"].dt.month - r["inner_origin"].dt.month
    assert (months == r["horizon"]).all()


def test_the_test_window_is_never_read(last_value_model):
    # The function is given only the training data. Adding later observations to
    # the full data and cutting at the outer origin gives the same residuals.
    full = _y(extra=12)
    full.loc[full["ds"] > OUTER, "y"] = -1e9            # poison the test window
    a = backtest_residuals("last_value", full[full["ds"] <= OUTER], H, N_INNER)
    b = backtest_residuals("last_value", _y(), H, N_INNER)
    pd.testing.assert_frame_equal(a, b)
    assert np.isfinite(a["resid"]).all() and a["y"].min() >= 0


def test_each_inner_forecast_sees_only_its_own_past(last_value_model):
    r = backtest_residuals("last_value", _y(), H, N_INNER)
    # The model forecasts the last value it saw and y rises by 1 per period, so
    # the h-step residual is exactly h. Any look-ahead would make it smaller.
    assert (r["resid"] == r["horizon"]).all()
    # context lengths: inner origin at position p has p + 1 observations
    lengths = [n for n, _ in last_value_model]
    assert lengths == list(range(T - H - 23, T - H + 1))
    assert len(last_value_model) == N_INNER
    # and the last value seen is the value at the inner origin, for the first series
    assert [last[0] for _, last in last_value_model] == [float(p - 1) for p in lengths]


def test_median_and_mean_arms(monkeypatch):
    def run(contexts, h, device):                       # right-skewed: mean above median
        last = np.array([c[-1] for c in contexts])
        q = np.array([0.0, 0, 0, 0, 0, 1, 2, 3, 12])    # median 0, mean 2
        return last[:, None, None] + np.zeros((1, h, 1)) + q

    monkeypatch.setitem(forecasters.MODELS, "skewed", (run, "none/skewed"))
    med = backtest_residuals("skewed", _y(), H, N_INNER, point="median")
    mean = backtest_residuals("skewed", _y(), H, N_INNER, point="mean")
    assert (med["resid"] == med["horizon"]).all()
    np.testing.assert_allclose(mean["resid"], mean["horizon"] - 2.0)
    np.testing.assert_allclose(mean["yhat"] - med["yhat"], 2.0)


def test_cache_per_dataset_origin_model_point(last_value_model, tmp_path):
    y = _y()
    a = backtest_residuals("last_value", y, H, N_INNER, point="median", dataset="Toy", cache_dir=tmp_path)
    paths = backtest_paths("Toy", OUTER, "last_value", tmp_path)
    assert paths["inner"].name == "backtest_Toy_2016-08-01_last_value.parquet"
    assert paths["residuals"]("median").name == "resid_Toy_2016-08-01_last_value_median.parquet"
    assert paths["inner"].exists() and paths["residuals"]("median").exists() and paths["timing"].exists()
    assert not paths["residuals"]("mean").exists()
    assert len(last_value_model) == N_INNER

    b = backtest_residuals("last_value", y, H, N_INNER, point="median", dataset="Toy", cache_dir=tmp_path)
    pd.testing.assert_frame_equal(a, b)
    # the mean arm reuses the cached inner forecasts: no new model calls
    backtest_residuals("last_value", y, H, N_INNER, point="mean", dataset="Toy", cache_dir=tmp_path)
    assert paths["residuals"]("mean").exists()
    assert len(last_value_model) == N_INNER

    timing = pd.read_parquet(paths["timing"]).iloc[0]
    assert timing["forecast_calls"] == N_INNER and timing["series"] == len(IDS)

    # another outer origin is another cache entry
    backtest_residuals("last_value", y[y["ds"] <= DATES[-13]], H, N_INNER, dataset="Toy", cache_dir=tmp_path)
    assert len(last_value_model) == 2 * N_INNER


def test_inner_forecasts_carry_the_quantiles(last_value_model):
    f = inner_forecasts("last_value", _y(), H, N_INNER)
    assert len(f) == len(IDS) * N_INNER * H
    np.testing.assert_allclose(f["q90"] - f["q10"], 0.8)
    assert f["y"].notna().all()


def test_residual_array_shape_and_order(last_value_model):
    r = backtest_residuals("last_value", _y(), H, N_INNER)
    shuffled = r.sample(frac=1, random_state=0)
    arr = residual_array(shuffled, ["a", "b", "c"])
    assert arr.shape == (3, N_INNER, H)
    # residual = horizon for every series and inner origin
    np.testing.assert_array_equal(arr, np.broadcast_to(np.arange(1, H + 1), (3, N_INNER, H)))

    # values land in the right cell: mark one residual
    r2 = r.copy()
    inner = np.sort(r2["inner_origin"].unique())
    hit = (r2.unique_id == "c") & (r2.inner_origin == inner[5]) & (r2.horizon == 7)
    r2.loc[hit, "resid"] = -99.0
    arr2 = residual_array(r2, ["a", "b", "c"])
    assert arr2[2, 5, 6] == -99.0
    assert (arr2 == -99.0).sum() == 1


def test_residual_array_raises_on_missing_series(last_value_model):
    r = backtest_residuals("last_value", _y(), H, N_INNER)
    with pytest.raises(ValueError):
        residual_array(r, ["a", "b", "c", "zzz"])
