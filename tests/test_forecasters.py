import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import forecasters
from src.forecasters import COLUMNS, QUANTILE_COLS, cache_paths, forecast

ROOT = Path(__file__).resolve().parents[1]


def _context(n_obs=48, ids=("b", "a", "c")):
    ds = pd.date_range("2010-01-01", periods=n_obs, freq="MS")
    rows = []
    for k, uid in enumerate(ids):
        y = 100 * (k + 1) + 10 * np.sin(2 * np.pi * np.arange(n_obs) / 12)
        rows.append(pd.DataFrame({"unique_id": uid, "ds": ds, "y": y}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def fake_model(monkeypatch):
    """A model whose quantiles are last value + level, and that counts its calls."""
    calls = []

    def run(contexts, h, device):
        calls.append(len(contexts))
        last = np.array([c[-1] for c in contexts])
        return last[:, None, None] + np.zeros((1, h, 1)) + np.array(forecasters.QUANTILE_LEVELS)

    monkeypatch.setitem(forecasters.MODELS, "fake", (run, "none/fake"))
    return calls


def test_forecast_format(fake_model):
    ctx = _context()
    out = forecast("fake", ctx, 6)
    assert list(out.columns) == COLUMNS
    assert len(out) == 3 * 6
    assert list(out["unique_id"].unique()) == ["b", "a", "c"]  # order of context_df is kept
    expected_ds = pd.date_range("2014-01-01", periods=6, freq="MS")
    for _, g in out.groupby("unique_id"):
        assert list(g["ds"]) == list(expected_ds)
    assert (out["yhat"] == out["q50"]).all()
    assert (np.diff(out[QUANTILE_COLS].to_numpy(), axis=1) > 0).all()


def test_forecast_values_belong_to_the_right_series(fake_model):
    ctx = _context()
    out = forecast("fake", ctx, 3)
    last = ctx.sort_values("ds").groupby("unique_id")["y"].last()
    for uid, g in out.groupby("unique_id"):
        np.testing.assert_allclose(g["q10"], last[uid] + 0.1)


def test_forecast_does_not_depend_on_row_order(fake_model):
    ctx = _context()
    shuffled = ctx.sample(frac=1, random_state=0)
    a = forecast("fake", ctx, 3).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    b = forecast("fake", shuffled, 3).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_forecast_is_cached_by_dataset_origin_model(fake_model, tmp_path):
    ctx = _context()
    first = forecast("fake", ctx, 6, dataset="Toy", cache_dir=tmp_path)
    base_path, timing_path = cache_paths("Toy", "2013-12-01", "fake", tmp_path)
    assert base_path.exists() and timing_path.exists()
    assert base_path.name == "base_Toy_2013-12-01_fake.parquet"
    second = forecast("fake", ctx, 6, dataset="Toy", cache_dir=tmp_path)
    assert fake_model == [3]  # the model ran once
    pd.testing.assert_frame_equal(first, second)

    # another origin is another cache entry
    forecast("fake", ctx[ctx["ds"] <= "2012-12-01"], 6, dataset="Toy", cache_dir=tmp_path)
    assert fake_model == [3, 3]
    assert cache_paths("Toy", "2012-12-01", "fake", tmp_path)[0].exists()


def test_forecast_without_dataset_is_not_cached(fake_model, tmp_path):
    forecast("fake", _context(), 6, cache_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_forecast_rejects_bad_input(fake_model):
    ctx = _context()
    with pytest.raises(ValueError, match="Unknown model"):
        forecast("nope", ctx, 6)
    with pytest.raises(ValueError, match="same timestamp"):
        forecast("fake", ctx.iloc[:-1], 6)
    bad = ctx.copy()
    bad.loc[5, "y"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        forecast("fake", bad, 6)


CHRONOS_CHECK = """
import numpy as np, pandas as pd
from src.forecasters import COLUMNS, QUANTILE_COLS, forecast

ds = pd.date_range("2010-01-01", periods=72, freq="MS")
ctx = pd.concat([
    pd.DataFrame({"unique_id": uid, "ds": ds,
                  "y": 100 * (k + 1) + 10 * np.sin(2 * np.pi * np.arange(72) / 12)})
    for k, uid in enumerate(["b", "a", "c"])
], ignore_index=True)
out = forecast("chronos_bolt_small", ctx, 12, freq="MS")
assert list(out.columns) == COLUMNS
assert len(out) == 3 * 12
assert list(out["ds"].iloc[:12]) == list(pd.date_range("2016-01-01", periods=12, freq="MS"))
assert np.isfinite(out[["yhat", *QUANTILE_COLS]].to_numpy()).all()
assert (out["yhat"] == out["q50"]).all()
assert (out["q90"] > out["q10"]).all()
# a clean seasonal series around 100, 200, 300: the median stays in a sane range
med = out.groupby("unique_id")["yhat"].mean()
assert 80 < med["b"] < 120 and 180 < med["a"] < 220 and 280 < med["c"] < 320, med
again = forecast("chronos_bolt_small", ctx, 12, freq="MS")
pd.testing.assert_frame_equal(out, again)
print("CHRONOS_OK")
"""


def test_chronos_bolt_small_runs():
    # In its own process: torch cannot share a process with hierarchicalforecast on
    # macOS (two OpenMP runtimes), and other test modules import hierarchicalforecast.
    proc = subprocess.run([sys.executable, "-c", CHRONOS_CHECK], cwd=ROOT, capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0 and "CHRONOS_OK" in proc.stdout, proc.stderr[-2000:]


def test_forecast_refuses_to_run_torch_next_to_hierarchicalforecast(monkeypatch):
    monkeypatch.setattr(forecasters.sys, "platform", "darwin")
    monkeypatch.setitem(forecasters.sys.modules, "hierarchicalforecast._lib", object())
    with pytest.raises(RuntimeError, match="OpenMP"):
        forecast("chronos_bolt_small", _context(), 6)
