import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import forecasters
from src.forecasters import COLUMNS, QUANTILE_COLS, cache_paths, forecast, point_forecast

ROOT = Path(__file__).resolve().parents[1]


def _context(n_obs=48, ids=("b", "a", "c")):
    ds = pd.date_range("2010-01-01", periods=n_obs, freq="MS")
    rows = []
    for k, uid in enumerate(ids):
        y = 100 * (k + 1) + 10 * np.sin(2 * np.pi * np.arange(n_obs) / 12)
        rows.append(pd.DataFrame({"unique_id": uid, "ds": ds, "y": y}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def fake_model(register_model):
    """A model whose quantiles are last value + level, and that counts its calls."""
    calls = []

    def run(contexts, h):
        calls.append(len(contexts))
        last = np.array([c[-1] for c in contexts])
        return last[:, None, None] + np.zeros((1, h, 1)) + np.array(forecasters.QUANTILE_LEVELS)

    register_model("fake", quantiles=run)
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


def test_point_forecast_median_and_mean():
    base = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 18.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 12.0]], columns=QUANTILE_COLS)
    np.testing.assert_allclose(point_forecast(base, "median"), [5.0, 0.0])
    np.testing.assert_allclose(point_forecast(base, "mean"), [6.0, 2.0])  # 54/9 and 18/9
    with pytest.raises(ValueError):
        point_forecast(base, "mode")


def test_point_forecast_median_is_yhat(fake_model):
    out = forecast("fake", _context(), 4)
    np.testing.assert_array_equal(point_forecast(out, "median"), out["yhat"].to_numpy())
    np.testing.assert_allclose(point_forecast(out, "mean"), out["q50"].to_numpy())  # symmetric fake
    np.testing.assert_array_equal(point_forecast(out, "mean"), out["mean"].to_numpy())


def test_mean9_is_the_average_of_the_nine_common_levels(register_model):
    levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    values = np.array([-50.0, -5, 1, 2, 3, 4, 5, 6, 7, 8, 18, 40, 400])
    register_model("wide", levels=levels,
                   quantiles=lambda contexts, h: np.zeros((len(contexts), h, 1)) + values)
    out = forecast("wide", _context(), 2)
    np.testing.assert_allclose(point_forecast(out, "mean9"), 54 / 9)            # 1 + ... + 8 + 18
    np.testing.assert_allclose(point_forecast(out, "mean"), values.mean())      # all 13 levels
    np.testing.assert_allclose(point_forecast(out, "median"), 5.0)
    assert not np.allclose(point_forecast(out, "mean"), point_forecast(out, "mean9"))


def test_mean9_equals_mean_for_a_nine_level_model(fake_model):
    out = forecast("fake", _context(), 3)
    np.testing.assert_allclose(point_forecast(out, "mean9"), point_forecast(out, "mean"))


def test_frames_cached_before_the_mean_column_still_work(fake_model):
    old = forecast("fake", _context(), 4).drop(columns=["mean"])
    np.testing.assert_allclose(point_forecast(old, "mean"), old[QUANTILE_COLS].mean(axis=1))
    restored = forecasters.with_mean(old)
    assert list(restored.columns) == COLUMNS
    np.testing.assert_allclose(restored["mean"], old[QUANTILE_COLS].mean(axis=1))


def test_quantile_model_with_more_native_levels(register_model):
    # 11 native levels: the mean averages all 11, the extra levels are kept as columns.
    levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    register_model("wide", levels=levels,
                   quantiles=lambda contexts, h: np.zeros((len(contexts), h, 1)) + 100 * np.array(levels) ** 2)
    out = forecast("wide", _context(), 3)
    assert list(out.columns) == COLUMNS + ["q5", "q95"]
    np.testing.assert_allclose(out["q5"], 0.25)
    np.testing.assert_allclose(out["q95"], 90.25)
    np.testing.assert_allclose(out["yhat"], 25.0)
    np.testing.assert_allclose(out["mean"], np.mean(100 * np.array(levels) ** 2))
    assert not np.allclose(out["mean"], out[QUANTILE_COLS].mean(axis=1))


def test_quantile_model_must_output_the_nine_levels(register_model):
    register_model("narrow", levels=[0.25, 0.5, 0.75],
                   quantiles=lambda contexts, h: np.zeros((len(contexts), h, 3)))
    with pytest.raises(RuntimeError, match="quantile levels"):
        forecast("narrow", _context(), 3)


def test_sample_model_median_mean_and_quantiles(register_model, tmp_path):
    def samples(contexts, h, seed):
        rng = np.random.default_rng(seed)
        return rng.lognormal(mean=2.0, sigma=1.0, size=(len(contexts), h, 100))

    register_model("sampler", samples=samples)
    ctx = _context()
    out = forecast("sampler", ctx, 3, dataset="Toy", cache_dir=tmp_path)
    assert list(out.columns) == COLUMNS + ["sample_var"]
    s = pd.read_parquet(forecasters.samples_path("Toy", "2013-12-01", "sampler", tmp_path))
    assert list(s.columns) == ["unique_id", "ds"] + [f"s{i}" for i in range(100)]
    assert len(s) == len(out) and list(s["unique_id"]) == list(out["unique_id"])
    draws = s[[f"s{i}" for i in range(100)]].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(out["yhat"], np.median(draws, axis=1), rtol=1e-6)
    np.testing.assert_allclose(out["mean"], draws.mean(axis=1), rtol=1e-6)
    np.testing.assert_allclose(out["q10"], np.quantile(draws, 0.1, axis=1), rtol=1e-6)
    np.testing.assert_allclose(out["q90"], np.quantile(draws, 0.9, axis=1), rtol=1e-6)
    np.testing.assert_allclose(out["sample_var"], draws.var(axis=1, ddof=1), rtol=1e-6)
    # a frame cached before the column existed gets it back from the stored sample paths
    base_path, _ = cache_paths("Toy", "2013-12-01", "sampler", tmp_path)
    pd.read_parquet(base_path).drop(columns="sample_var").to_parquet(base_path, index=False)
    restored = forecast("sampler", ctx, 3, dataset="Toy", cache_dir=tmp_path)
    np.testing.assert_allclose(restored["sample_var"], draws.var(axis=1, ddof=1), rtol=1e-6)
    assert (out["mean"] > out["yhat"]).mean() > 0.9        # right-skewed
    # the seed is fixed per origin: the same call gives the same samples
    again = forecast("sampler", ctx, 3)
    pd.testing.assert_frame_equal(out, again)
    other = forecast("sampler", ctx[ctx["ds"] <= "2012-12-01"], 3)
    assert not np.allclose(other["yhat"], out["yhat"])


def test_forecast_cutoffs_uses_only_the_past(fake_model):
    ctx = _context()
    cutoffs = [pd.Timestamp("2012-06-01"), pd.Timestamp("2013-01-01")]
    out, info = forecasters.forecast_cutoffs("fake", ctx, cutoffs, 2)
    assert list(out["cutoff"].unique()) == cutoffs
    assert len(info["seconds"]) == 2 and fake_model == [3, 3]
    for cutoff, g in out.groupby("cutoff"):
        assert g["ds"].min() > cutoff
        last = ctx[ctx["ds"] == cutoff].set_index("unique_id")["y"]
        np.testing.assert_allclose(g["q10"], g["unique_id"].map(last) + 0.1)


def test_registry_is_pinned():
    from src.runners import FAMILIES, MODELS
    expected = {"chronos_bolt_tiny", "chronos_bolt_mini", "chronos_bolt_small", "chronos_t5_small",
                "chronos_2", "tirex", "moirai_2_small"}
    assert expected <= set(MODELS)
    for name in expected:
        spec = MODELS[name]
        assert len(spec["revision"]) == 40 and spec["family"] in FAMILIES
        assert spec["output"] in ("quantiles", "samples") and spec["env"]
    assert MODELS["tirex"]["repo"] == "NX-AI/TiRex"        # not the 1.1 checkpoint
    assert MODELS["chronos_t5_small"]["output"] == "samples"


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
    from src import runners
    monkeypatch.setattr(runners.sys, "platform", "darwin")
    monkeypatch.setitem(runners.sys.modules, "hierarchicalforecast._lib", object())
    monkeypatch.setattr(runners, "_LOADED", {})
    with pytest.raises(RuntimeError, match="OpenMP"):
        forecast("chronos_bolt_small", _context(), 6)
