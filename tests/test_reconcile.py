"""PREREG §8.2: the numpy reconciler matches hierarchicalforecast to 1e-8.

The comparison uses the cached AutoETS forecasts and fitted values for the
2015-12 origin of TourismLarge, written by notebooks/02_gate_ets.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hierarchicalforecast.methods import MinTrace
from hierarchicalforecast.utils import (
    _shrunk_covariance_schaferstrimmer_no_nans,
    _shrunk_covariance_schaferstrimmer_with_nans,
)

from src.data import load_hierarchy
from src.reconcile import (
    PV_FLOOR_FACTOR, W_hybrid, W_ols, W_pv, W_shrink, W_shrink_bt, W_struct, W_var, W_var_bt,
    predictive_variance, projection_matrix, pv_floor, reconcile, reconcile_by_horizon,
)

TOL = 1e-8
RESULTS = Path(__file__).resolve().parents[1] / "results"
KEY = "TourismLarge_2015-12-01_AutoETS"
METHODS = ["ols", "wls_struct", "wls_var", "mint_shrink"]


# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def ets():
    rec_path = RESULTS / f"rec_{KEY}_hf.parquet"
    fitted_path = RESULTS / f"fitted_{KEY}.parquet"
    if not (rec_path.exists() and fitted_path.exists()):
        pytest.fail(f"Cached forecasts not found in {RESULTS}. Run notebooks/02_gate_ets.py first.")

    _, S_df, _, _, _ = load_hierarchy("TourismLarge")
    ids = S_df["unique_id"].to_numpy()
    bottom_ids = [c for c in S_df.columns if c != "unique_id"]
    rec = pd.read_parquet(rec_path)
    fitted = pd.read_parquet(fitted_path)

    def wide(df, col):
        return df.pivot(index="unique_id", columns="ds", values=col).loc[ids].to_numpy(dtype=np.float64)

    S = S_df[bottom_ids].to_numpy(dtype=np.float64)
    y_insample, y_fitted = wide(fitted, "y"), wide(fitted, "AutoETS")
    out = {
        "S": S,
        "y_hat": wide(rec, "AutoETS"),
        "y_insample": y_insample,
        "y_fitted": y_fitted,
        "resid": y_insample - y_fitted,
        "hf_cached": {m: wide(rec, f"AutoETS/MinTrace_method-{m}") for m in METHODS},
    }
    assert out["y_hat"].shape == (555, 12)
    assert out["resid"].shape == (555, 216)
    assert not np.isnan(out["resid"]).any()
    return out


@pytest.fixture(scope="module")
def hf(ets):
    """hierarchicalforecast run directly on the same arrays: y_tilde and W."""
    out = {}
    for m in METHODS:
        rec = MinTrace(method=m, nonnegative=False)
        y_tilde = rec.fit_predict(S=ets["S"], y_hat=ets["y_hat"], y_insample=ets["y_insample"],
                                  y_hat_insample=ets["y_fitted"])["mean"]
        out[m] = {"y_tilde": y_tilde, "W": rec.W}
    return out


@pytest.fixture(scope="module")
def ours(ets):
    S, resid = ets["S"], ets["resid"]
    W_s, lam = W_shrink(resid, return_lambda=True)
    return {
        "ols": W_ols(S.shape[0]),
        "wls_struct": W_struct(S),
        "wls_var": W_var(resid),
        "mint_shrink": W_s,
        "lambda": lam,
    }


# ------------------------------------------- gate: match hierarchicalforecast

@pytest.mark.parametrize("method", METHODS)
def test_y_tilde_matches_hierarchicalforecast_cached(ets, ours, method):
    y_tilde = reconcile(ets["y_hat"], ets["S"], ours[method])
    assert np.abs(y_tilde - ets["hf_cached"][method]).max() <= TOL


@pytest.mark.parametrize("method", METHODS)
def test_y_tilde_matches_hierarchicalforecast_direct_call(ets, hf, ours, method):
    y_tilde = reconcile(ets["y_hat"], ets["S"], ours[method])
    assert np.abs(y_tilde - hf[method]["y_tilde"]).max() <= TOL


@pytest.mark.parametrize("method", METHODS)
def test_W_matches_hierarchicalforecast(hf, ours, method):
    # Compared on the correlation scale, |dW_ij| / sqrt(W_ii W_jj). A plain relative
    # tolerance fails on covariances that are near zero by cancellation.
    W_hf = hf[method]["W"]
    scale = np.sqrt(np.outer(np.diag(W_hf), np.diag(W_hf)))
    assert (np.abs(ours[method] - W_hf) / scale).max() <= 1e-12


def test_lambda_matches_hierarchicalforecast(ets, hf, ours):
    # The library does not expose lambda. Its off-diagonal W is (1 - lambda) * cov,
    # so lambda is backed out from the ratio, which must be one constant.
    cov = np.cov(ets["resid"])
    low = np.tril_indices(cov.shape[0], k=-1)
    ratio = hf["mint_shrink"]["W"][low] / cov[low]
    lam_hf = 1.0 - np.median(ratio)
    assert np.abs(ratio - np.median(ratio)).max() <= 1e-9
    assert 0.0 < ours["lambda"] < 1.0
    assert abs(ours["lambda"] - lam_hf) <= 1e-12
    # diagonal is the unshrunk sample variance
    np.testing.assert_allclose(np.diag(hf["mint_shrink"]["W"]), np.diag(cov), rtol=1e-12)
    np.testing.assert_allclose(np.diag(ours["mint_shrink"]), np.diag(cov), rtol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_coherence(ets, ours, method):
    S = ets["S"]
    y_tilde = reconcile(ets["y_hat"], S, ours[method])
    bottom = y_tilde[-S.shape[1]:]
    assert np.abs(S @ bottom - y_tilde).max() <= TOL


@pytest.mark.parametrize("method", METHODS)
def test_SGS_equals_S(ets, ours, method):
    S = ets["S"]
    G = projection_matrix(S, ours[method])
    assert G.shape == (S.shape[1], S.shape[0])
    assert np.abs(S @ G @ S - S).max() <= TOL
    assert np.abs(G @ S - np.eye(S.shape[1])).max() <= TOL


@pytest.mark.parametrize("method", METHODS)
def test_coherent_forecasts_are_unchanged(ets, ours, method):
    S = ets["S"]
    coherent = S @ ets["y_hat"][-S.shape[1]:]
    assert np.abs(reconcile(coherent, S, ours[method]) - coherent).max() <= TOL


# ------------------------------------------- hand-built hierarchy: T = A + B

S3 = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
Y3 = np.array([10.0, 3.0, 4.0])  # incoherent by d = 10 - (3 + 4) = 3


def test_hand_ols():
    # Equal weights: the gap of 3 is split equally. T -1, A +1, B +1.
    np.testing.assert_allclose(reconcile(Y3, S3, W_ols(3)), [9.0, 4.0, 5.0], atol=1e-12)


def test_hand_wls_struct():
    # W = diag(2, 1, 1): each series moves by w_i * d / sum(w) = w_i * 3 / 4.
    np.testing.assert_allclose(W_struct(S3), np.diag([2.0, 1.0, 1.0]))
    np.testing.assert_allclose(reconcile(Y3, S3, W_struct(S3)), [8.5, 3.75, 4.75], atol=1e-12)


def test_hand_wls_unequal_variances():
    # W = diag(1, 2, 3): moves are 3 * (1, 2, 3) / 6 = 0.5, 1, 1.5.
    W = np.diag([1.0, 2.0, 3.0])
    np.testing.assert_allclose(reconcile(Y3, S3, W), [9.5, 4.0, 5.5], atol=1e-12)


def test_hand_full_covariance():
    # y_tilde = y_hat - W U (U'WU)^-1 U'y_hat with U' = [1, -1, -1].
    # W U = [2, -1, -2], U'WU = 5, U'y_hat = 3  ->  y_hat - [2, -1, -2] * 3/5.
    W = np.array([[4.0, 1.0, 1.0], [1.0, 2.0, 0.0], [1.0, 0.0, 3.0]])
    np.testing.assert_allclose(reconcile(Y3, S3, W), [8.8, 3.6, 5.2], atol=1e-12)


def test_hand_projection_matrix_ols():
    # (S'S)^-1 S' with S'S = [[2, 1], [1, 2]]
    G = projection_matrix(S3, W_ols(3))
    np.testing.assert_allclose(G, np.array([[1.0, 2.0, -1.0], [1.0, -1.0, 2.0]]) / 3, atol=1e-12)


def test_hand_multiple_horizons_are_reconciled_independently():
    y_hat = np.column_stack([Y3, [7.0, 3.0, 4.0], [20.0, 8.0, 6.0]])
    out = reconcile(y_hat, S3, W_ols(3))
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out[:, 0], [9.0, 4.0, 5.0], atol=1e-12)
    np.testing.assert_allclose(out[:, 1], [7.0, 3.0, 4.0], atol=1e-12)   # already coherent
    np.testing.assert_allclose(out[:, 2], [18.0, 10.0, 8.0], atol=1e-12)  # gap 6 -> -2, +2, +2


def test_scaling_W_does_not_change_the_result():
    W = np.array([[4.0, 1.0, 1.0], [1.0, 2.0, 0.0], [1.0, 0.0, 3.0]])
    np.testing.assert_allclose(reconcile(Y3, S3, 1000.0 * W), reconcile(Y3, S3, W), atol=1e-12)


def test_reconcile_rejects_bad_input():
    with pytest.raises(ValueError):
        reconcile(Y3, S3, np.eye(2))
    with pytest.raises(ValueError):
        reconcile(Y3[:2], S3, np.eye(3))
    with pytest.raises(ValueError):
        reconcile(np.array([10.0, np.nan, 4.0]), S3, np.eye(3))
    with pytest.raises(ValueError):
        reconcile(Y3, S3, np.diag([1.0, np.nan, 1.0]))


# ------------------------------------------------------- W builders by hand

def test_W_var_hand_computed_is_not_centered():
    # Row means are not zero. Mean of squares: (1+1+4)/3 = 2 and (0+0+9)/3 = 3.
    resid = np.array([[1.0, -1.0, 2.0], [0.0, 0.0, 3.0]])
    np.testing.assert_allclose(np.diag(W_var(resid)), [2.0 + 2e-8, 3.0 + 2e-8], rtol=1e-15)
    assert np.count_nonzero(W_var(resid) - np.diag(np.diag(W_var(resid)))) == 0


def test_W_var_nan_is_skipped_but_counted_in_T():
    resid = np.array([[1.0, np.nan, 2.0]])
    np.testing.assert_allclose(np.diag(W_var(resid)), [5.0 / 3 + 2e-8], rtol=1e-15)


def test_W_var_zero_residuals_give_the_jitter():
    np.testing.assert_allclose(np.diag(W_var(np.zeros((2, 5)))), [2e-8, 2e-8])


def test_W_shrink_uncorrelated_series_shrink_fully():
    # Sample correlation is 0, so lambda clips to 1 and W is the diagonal, var = 4/3.
    resid = np.array([[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0]])
    W, lam = W_shrink(resid, return_lambda=True)
    assert lam == 1.0
    np.testing.assert_allclose(W, np.diag([4 / 3, 4 / 3]), atol=1e-12)


def test_W_shrink_perfectly_correlated_series_do_not_shrink():
    # y = 2x: the standardised products are constant, so lambda = 0 and W = cov.
    x = np.array([1.0, -1.0, 1.0, -1.0])
    W, lam = W_shrink(np.vstack([x, 2 * x]), return_lambda=True)
    assert lam == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(W, np.array([[4 / 3, 8 / 3], [8 / 3, 16 / 3]]), atol=1e-7)


def test_W_shrink_hand_computed_lambda():
    # x = (1,-1,1,-1), y = (1,-1,0,0): c_k = (1,1,0,0), sd_x = 1, sd_y = sqrt(1/2).
    # w_k = sqrt(2) * (1,1,0,0), wbar = sqrt(2)/2, sum (w - wbar)^2 = 2.
    # lambda = [2 / (4*3)] / [1/2] = 1/3. cov_xy = 2/3, so W_xy = (2/3) * (2/3) = 4/9.
    resid = np.array([[1.0, -1.0, 1.0, -1.0], [1.0, -1.0, 0.0, 0.0]])
    W, lam = W_shrink(resid, return_lambda=True)
    assert lam == pytest.approx(1 / 3, abs=1e-6)  # the library's 2e-8 terms move it slightly
    np.testing.assert_allclose(W, np.array([[4 / 3, 4 / 9], [4 / 9, 2 / 3]]), atol=1e-6)


def test_W_shrink_is_centered():
    rng = np.random.default_rng(0)
    resid = rng.normal(size=(5, 40))
    np.testing.assert_allclose(W_shrink(resid + 100.0), W_shrink(resid), rtol=1e-9)


def test_W_shrink_diagonal_is_floored_at_ridge():
    rng = np.random.default_rng(1)
    resid = rng.normal(size=(4, 30))
    resid[2] = 0.0
    W = W_shrink(resid, ridge=0.5)
    assert W[2, 2] == 0.5
    assert np.isfinite(W).all()


def test_W_shrink_matches_library_function_without_nans():
    rng = np.random.default_rng(2)
    resid = rng.normal(size=(30, 50)) * rng.uniform(0.1, 100, size=(30, 1))
    resid[:10] += 0.5 * resid[10:20]
    expected = _shrunk_covariance_schaferstrimmer_no_nans(resid, 2e-8)
    W, lam = W_shrink(resid, return_lambda=True)
    assert 0.0 < lam < 1.0
    np.testing.assert_allclose(W, expected, rtol=1e-12, atol=1e-14)


def test_W_shrink_matches_library_function_with_nans():
    rng = np.random.default_rng(3)
    resid = rng.normal(size=(30, 50)) * rng.uniform(0.1, 100, size=(30, 1))
    resid[:10] += 0.5 * resid[10:20]
    resid[rng.random(resid.shape) < 0.1] = np.nan
    resid[0, :12] = np.nan  # a leading block, as from a fitted-values warm-up
    expected = _shrunk_covariance_schaferstrimmer_with_nans(resid, ~np.isnan(resid), 2e-8)
    W, lam = W_shrink(resid, return_lambda=True)
    assert 0.0 < lam < 1.0
    np.testing.assert_allclose(W, expected, rtol=1e-12, atol=1e-14)




# ------------------------------------------------------------- W_pv (WLS-pv)

def _quantiles(q10, q90):
    """(n_series, 9) with the given q10 and q90 and a straight line between."""
    q10, q90 = np.asarray(q10, dtype=float), np.asarray(q90, dtype=float)
    return q10[:, None] + (q90 - q10)[:, None] * np.linspace(0, 1, 9)[None, :]


def test_predictive_variance_hand_computed():
    # width 2 * 1.2816 -> v = 1 ; width 3 * 2 * 1.2816 -> v = 9 ; location does not matter
    q = _quantiles([0.0, 10.0, -5.0], [2.5632, 10.0 + 7.6896, -5.0 + 1.2816])
    np.testing.assert_allclose(predictive_variance(q), [1.0, 9.0, 0.25], rtol=1e-12)


def test_predictive_variance_uses_only_q10_and_q90():
    q = _quantiles([0.0], [2.5632])
    q2 = q.copy()
    q2[:, 1:-1] = 99.0
    np.testing.assert_allclose(predictive_variance(q2), predictive_variance(q))


def test_predictive_variance_recovers_a_normal_variance():
    from scipy.stats import norm

    levels = np.arange(0.1, 1.0, 0.1)
    sigma = np.array([0.5, 2.0, 30.0])
    q = norm.ppf(levels[None, :], loc=np.array([3.0, -1.0, 100.0])[:, None], scale=sigma[:, None])
    # 1.2816 is the 4-decimal rounding of 1.28155..., hence the 1e-4 tolerance
    np.testing.assert_allclose(predictive_variance(q), sigma ** 2, rtol=1e-4)


def test_W_pv_is_diagonal_with_the_variances():
    q = _quantiles([0.0, 10.0, -5.0], [2.5632, 17.6896, -3.7184])
    W = W_pv(q)
    assert W.shape == (3, 3)
    np.testing.assert_allclose(np.diag(W), [1.0, 9.0, 0.25], rtol=1e-12)
    assert np.count_nonzero(W - np.diag(np.diag(W))) == 0


def test_W_pv_per_horizon():
    q1 = _quantiles([0.0, 0.0, 0.0], [2.5632, 2.5632, 2.5632])          # v = 1, 1, 1
    q2 = _quantiles([0.0, 0.0, 0.0], [2.5632, 2 * 2.5632, 3 * 2.5632])  # v = 1, 4, 9
    Ws = W_pv(np.stack([q1, q2], axis=1))  # (n_series, h=2, 9)
    assert len(Ws) == 2
    np.testing.assert_allclose(np.diag(Ws[0]), [1.0, 1.0, 1.0], rtol=1e-12)
    np.testing.assert_allclose(np.diag(Ws[1]), [1.0, 4.0, 9.0], rtol=1e-12)


def test_W_pv_raises_on_zero_or_negative_width():
    with pytest.raises(ValueError, match="q90 <= q10"):
        W_pv(_quantiles([0.0, 1.0], [1.0, 1.0]))
    with pytest.raises(ValueError, match="q90 <= q10"):
        W_pv(_quantiles([0.0, 1.0], [1.0, 0.5]))


def test_W_pv_rejects_wrong_shape_and_nan():
    with pytest.raises(ValueError):
        W_pv(np.ones((3, 5)))
    q = _quantiles([0.0], [1.0])
    q[0, 0] = np.nan
    with pytest.raises(ValueError):
        W_pv(q)


def test_hand_wls_pv():
    # Variances 1, 2, 3 -> same answer as test_hand_wls_unequal_variances.
    s = np.sqrt([1.0, 2.0, 3.0])
    q = _quantiles(Y3 - 1.2816 * s, Y3 + 1.2816 * s)
    np.testing.assert_allclose(np.diag(W_pv(q)), [1.0, 2.0, 3.0], rtol=1e-12)
    np.testing.assert_allclose(reconcile(Y3, S3, W_pv(q)), [9.5, 4.0, 5.5], atol=1e-12)


def test_reconcile_by_horizon_uses_each_horizons_W():
    y_hat = np.column_stack([Y3, Y3])
    Ws = [np.diag([1.0, 1.0, 1.0]), np.diag([1.0, 2.0, 3.0])]
    out = reconcile_by_horizon(y_hat, S3, Ws)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[:, 0], [9.0, 4.0, 5.0], atol=1e-12)
    np.testing.assert_allclose(out[:, 1], [9.5, 4.0, 5.5], atol=1e-12)


def test_reconcile_by_horizon_with_one_W_equals_reconcile(ets, ours):
    S, y_hat = ets["S"], ets["y_hat"]
    W = ours["wls_var"]
    out = reconcile_by_horizon(y_hat, S, [W] * y_hat.shape[1])
    assert np.abs(out - reconcile(y_hat, S, W)).max() <= TOL


def test_reconcile_by_horizon_needs_one_W_per_horizon():
    with pytest.raises(ValueError):
        reconcile_by_horizon(np.column_stack([Y3, Y3]), S3, [np.eye(3)])


# ------------------------------------------ backtest W builders and the Hybrid

def _resid3(n=6, n_inner=24, h=3, seed=0):
    """(n_series, n_inner, h) with correlated series and a scale that grows with h."""
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(1, n_inner, h))
    r = rng.normal(size=(n, n_inner, h)) + 1.5 * common
    return r * rng.uniform(0.5, 20, size=(n, 1, 1)) * np.arange(1, h + 1)


def test_W_var_bt_is_W_var_per_horizon():
    r = _resid3()
    Ws = W_var_bt(r)
    assert len(Ws) == 3
    for t, W in enumerate(Ws):
        np.testing.assert_array_equal(W, W_var(r[:, :, t]))
    # uncentered: adding a bias raises the variance by bias^2
    biased = W_var_bt(r + 10.0)
    assert (np.diag(biased[0]) > np.diag(Ws[0])).all()


def test_W_var_bt_hand_computed():
    # 1 series, 2 inner origins, 2 horizons. h=1 residuals (1, -1), h=2 residuals (3, 5).
    r = np.array([[[1.0, 3.0], [-1.0, 5.0]]])
    Ws = W_var_bt(r)
    np.testing.assert_allclose(Ws[0], [[1.0 + 2e-8]])
    np.testing.assert_allclose(Ws[1], [[17.0 + 2e-8]])     # (9 + 25) / 2, not the variance 1


def test_W_shrink_bt_is_W_shrink_per_horizon():
    r = _resid3()
    Ws, lams = W_shrink_bt(r, return_lambda=True)
    assert len(Ws) == len(lams) == 3
    for t in range(3):
        W, lam = W_shrink(r[:, :, t], return_lambda=True)
        np.testing.assert_array_equal(Ws[t], W)
        assert lams[t] == lam
    assert len(W_shrink_bt(r)) == 3


def test_backtest_builders_need_three_dimensions():
    with pytest.raises(ValueError):
        W_var_bt(np.ones((4, 24)))
    with pytest.raises(ValueError):
        W_shrink_bt(np.ones((4, 24)))


def _q_with_variance(v):
    s = np.sqrt(np.asarray(v, dtype=float))
    return _quantiles(-1.2816 * s, 1.2816 * s)


def test_W_hybrid_is_symmetric_positive_definite():
    r = _resid3(n=8, n_inner=24, h=1)[:, :, 0]
    q = _q_with_variance(np.linspace(1.0, 50.0, 8))
    W, lam = W_hybrid(q, r, return_lambda=True)
    assert 0.0 < lam < 1.0
    np.testing.assert_allclose(W, W.T, rtol=0, atol=1e-12)
    assert np.linalg.eigvalsh(W).min() > 0
    np.linalg.cholesky(W)


def test_W_hybrid_is_positive_definite_with_fewer_residuals_than_series():
    rng = np.random.default_rng(5)
    r = rng.normal(size=(40, 24)) + rng.normal(size=(1, 24))   # 40 series, 24 residuals
    W, lam = W_hybrid(_q_with_variance(rng.uniform(1, 9, size=40)), r, return_lambda=True)
    assert lam > 0
    assert np.linalg.matrix_rank(np.corrcoef(r)) < 40           # the raw correlation is singular
    assert np.linalg.eigvalsh(W).min() > 0


def test_W_hybrid_diagonal_is_the_predictive_variance():
    r = _resid3(n=5, h=1)[:, :, 0]
    v = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    W = W_hybrid(_q_with_variance(v), r)
    np.testing.assert_allclose(np.diag(W), v, rtol=1e-12)
    # and it does not depend on the scale of the residuals
    np.testing.assert_allclose(W_hybrid(_q_with_variance(v), r * np.array([[1], [10], [100], [3], [7]])),
                               W, rtol=1e-6)


def test_W_hybrid_correlations_are_the_shrunk_residual_correlations():
    r = _resid3(n=5, h=1)[:, :, 0]
    v = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    W, lam = W_hybrid(_q_with_variance(v), r, return_lambda=True)
    assert lam == W_shrink(r, return_lambda=True)[1]
    expected = (1 - lam) * np.corrcoef(r)
    expected[np.diag_indices(5)] = 1.0
    d = np.sqrt(v)
    np.testing.assert_allclose(W / np.outer(d, d), expected, rtol=1e-9, atol=1e-12)


def test_W_hybrid_reduces_to_W_pv_when_lambda_is_one():
    # Uncorrelated residuals: lambda clips to 1 (see test_W_shrink_uncorrelated_...).
    r = np.array([[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0]])
    q = _q_with_variance([3.0, 7.0])
    W, lam = W_hybrid(q, r, return_lambda=True)
    assert lam == 1.0
    np.testing.assert_allclose(W, W_pv(q), rtol=0, atol=1e-12)


def test_W_hybrid_hand_computed():
    # Residuals of test_W_shrink_hand_computed_lambda: lambda = 1/3, residual
    # correlation = (2/3) / sqrt(4/3 * 2/3) = 1/sqrt(2). Shrunk: (2/3) / sqrt(2).
    # With variances 4 and 9: W_12 = 2 * 3 * (2/3) / sqrt(2) = 4 / sqrt(2).
    r = np.array([[1.0, -1.0, 1.0, -1.0], [1.0, -1.0, 0.0, 0.0]])
    W, lam = W_hybrid(_q_with_variance([4.0, 9.0]), r, return_lambda=True)
    assert lam == pytest.approx(1 / 3, abs=1e-6)
    np.testing.assert_allclose(W, [[4.0, 4 / np.sqrt(2)], [4 / np.sqrt(2), 9.0]], atol=1e-5)


def test_W_hybrid_series_with_constant_residuals_is_uncorrelated():
    r = _resid3(n=4, h=1)[:, :, 0]
    r[2] = 0.0
    W = W_hybrid(_q_with_variance([1.0, 2.0, 3.0, 4.0]), r)
    assert np.isfinite(W).all()
    assert W[2, 2] == pytest.approx(3.0)
    assert np.count_nonzero(np.delete(W[2], 2)) == 0
    assert np.linalg.eigvalsh(W).min() > 0


def test_W_hybrid_per_horizon():
    r = _resid3(n=5, n_inner=24, h=3)
    q = np.stack([_q_with_variance(np.full(5, float(t + 1))) for t in range(3)], axis=1)  # (5, 3, 9)
    Ws, lams = W_hybrid(q, r, return_lambda=True)
    assert len(Ws) == len(lams) == 3
    for t in range(3):
        np.testing.assert_array_equal(Ws[t], W_hybrid(q[:, t, :], r[:, :, t]))
        np.testing.assert_allclose(np.diag(Ws[t]), t + 1.0, rtol=1e-12)
    with pytest.raises(ValueError):
        W_hybrid(q[:, :2, :], r)


def test_W_hybrid_raises_on_zero_interval_width():
    with pytest.raises(ValueError, match="q90 <= q10"):
        W_hybrid(_quantiles([0.0, 1.0], [1.0, 1.0]), np.ones((2, 5)))


@pytest.mark.parametrize("builder", ["var_bt", "shrink_bt", "hybrid"])
def test_backtest_W_reconciles_coherently(builder):
    rng = np.random.default_rng(7)
    r = rng.normal(size=(3, 24, 2)) * np.array([3.0, 1.0, 2.0])[:, None, None]
    q = np.stack([_q_with_variance([9.0, 1.0, 4.0])] * 2, axis=1)
    Ws = {"var_bt": W_var_bt(r), "shrink_bt": W_shrink_bt(r), "hybrid": W_hybrid(q, r)}[builder]
    out = reconcile_by_horizon(np.column_stack([Y3, Y3]), S3, Ws)
    np.testing.assert_allclose(out[0], out[1] + out[2], atol=1e-10)
    for W in Ws:
        G = projection_matrix(S3, W)
        np.testing.assert_allclose(S3 @ G @ S3, S3, atol=1e-10)


# ------------------------------------------------ predictive-variance floor

def test_pv_floor_is_the_registered_fraction_of_rmsse_scale():
    assert PV_FLOOR_FACTOR == 1e-4
    np.testing.assert_allclose(pv_floor([4.0, 0.0, 2.5e6]), [4e-4, 0.0, 250.0])


def test_W_pv_floor_hand_computed():
    # v = 1, 0 and 1e-6; floors 0.5, 0.25, 0.01 -> 1 (kept), 0.25 and 0.01 (floored)
    q = _quantiles([0.0, 5.0, 0.0], [2.5632, 5.0, 2.5632e-3])
    np.testing.assert_allclose(predictive_variance(q), [1.0, 0.0, 1e-6], rtol=1e-12, atol=0)
    W = W_pv(q, floor=np.array([0.5, 0.25, 0.01]))
    np.testing.assert_allclose(np.diag(W), [1.0, 0.25, 0.01], rtol=1e-12)
    assert np.count_nonzero(W - np.diag(np.diag(W))) == 0


def test_W_pv_floor_does_not_change_variances_above_it():
    q = _q_with_variance([1.0, 4.0, 9.0])
    np.testing.assert_array_equal(W_pv(q, floor=pv_floor([1.0, 1.0, 1.0])), W_pv(q))


def test_W_pv_floor_makes_a_zero_width_usable():
    q = _quantiles([0.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="q90 <= q10"):
        W_pv(q)
    W = W_pv(q, floor=pv_floor([100.0, 100.0]))
    assert W[1, 1] == pytest.approx(0.01)
    assert np.linalg.eigvalsh(W).min() > 0


def test_W_pv_floor_of_zero_cannot_rescue_a_zero_variance():
    # rmsse_scale 0 means a perfectly seasonal series: the floor is 0 too
    with pytest.raises(ValueError, match="zero"):
        W_pv(_quantiles([0.0, 1.0], [1.0, 1.0]), floor=pv_floor([1.0, 0.0]))


def test_W_pv_floor_per_horizon_is_per_series():
    q = np.stack([_q_with_variance([1.0, 1e-8]), _q_with_variance([4.0, 1.0])], axis=1)   # (2, h=2, 9)
    Ws = W_pv(q, floor=np.array([2.0, 0.5]))
    np.testing.assert_allclose(np.diag(Ws[0]), [2.0, 0.5], rtol=1e-12)
    np.testing.assert_allclose(np.diag(Ws[1]), [4.0, 1.0], rtol=1e-12)
    with pytest.raises(ValueError, match="floor must have shape"):
        W_pv(q, floor=np.array([2.0, 0.5, 1.0]))


def test_W_pv_from_a_variance():
    # sample-path models give the sample variance directly
    np.testing.assert_allclose(np.diag(W_pv(variance=[2.0, 3.0])), [2.0, 3.0])
    np.testing.assert_allclose(np.diag(W_pv(variance=[2.0, 0.0], floor=[1.0, 0.5])), [2.0, 0.5])
    Ws = W_pv(variance=np.array([[1.0, 2.0], [0.0, 4.0]]), floor=[0.1, 0.3])
    np.testing.assert_allclose(np.diag(Ws[0]), [1.0, 0.3])
    np.testing.assert_allclose(np.diag(Ws[1]), [2.0, 4.0])
    with pytest.raises(ValueError, match="zero"):
        W_pv(variance=[2.0, 0.0])
    with pytest.raises(ValueError, match="negative"):
        W_pv(variance=[2.0, -1.0])
    with pytest.raises(ValueError, match="either"):
        W_pv()
    with pytest.raises(ValueError, match="either"):
        W_pv(_q_with_variance([1.0]), variance=[1.0])


def test_W_hybrid_uses_the_floor_and_stays_positive_definite():
    r = _resid3(n=4, h=1)[:, :, 0]
    q = _quantiles([0.0, 0.0, 3.0, 0.0], [2.5632, 2.5632, 3.0, 2.5632])      # third has zero width
    with pytest.raises(ValueError):
        W_hybrid(q, r)
    W, lam = W_hybrid(q, r, return_lambda=True, floor=np.array([0.1, 0.1, 0.04, 0.1]))
    np.testing.assert_allclose(np.diag(W), [1.0, 1.0, 0.04, 1.0], rtol=1e-12)
    assert np.linalg.eigvalsh(W).min() > 0
    # the correlations are untouched by the floor
    expected = (1 - lam) * np.corrcoef(r)
    expected[np.diag_indices(4)] = 1.0
    d = np.sqrt(np.diag(W))
    np.testing.assert_allclose(W / np.outer(d, d), expected, rtol=1e-9, atol=1e-12)


def test_W_hybrid_from_a_sample_variance():
    r = _resid3(n=3, h=2)
    v = np.array([[1.0, 2.0], [0.0, 3.0], [4.0, 5.0]])
    Ws = W_hybrid(resid=r, variance=v, floor=np.array([0.5, 0.5, 0.5]))
    np.testing.assert_allclose(np.diag(Ws[0]), [1.0, 0.5, 4.0], rtol=1e-12)
    np.testing.assert_allclose(np.diag(Ws[1]), [2.0, 3.0, 5.0], rtol=1e-12)
    for W in Ws:
        assert np.linalg.eigvalsh(W).min() > 0


def test_floored_W_reconciles_coherently():
    q = _quantiles(Y3 - 1.0, [Y3[0] + 1.0, Y3[1] - 1.0, Y3[2] + 1.0])          # middle has zero width
    out = reconcile(Y3, S3, W_pv(q, floor=pv_floor([1.0, 1.0, 1.0])))
    assert out[0] == pytest.approx(out[1] + out[2])
    # the series with the floored, tiny variance is almost kept as it is
    assert abs(out[1] - Y3[1]) < 1e-3


# ------------------------------------- floor for M5 items with no sale yet

def _m5_case():
    #          store  item  item  item  item(unsold)  item(unsold)
    scale = np.array([900.0, 1.0, 4.0, 10.0, np.nan, np.nan])
    items = np.array([False, True, True, True, True, True])
    unsold = np.array([False, False, False, False, True, True])
    return scale, unsold, items


def test_pv_floor_unsold_items_use_the_median_scale_of_sold_items():
    scale, unsold, items = _m5_case()
    floor = pv_floor(scale, unsold=unsold, items=items)
    # median of the sold items' scales (1, 4, 10) is 4; the store's 900 is not an item
    np.testing.assert_allclose(floor, [0.09, 1e-4, 4e-4, 1e-3, 4e-4, 4e-4])


def test_pv_floor_median_is_over_sold_items_only():
    scale = np.array([900.0, 2.0, np.nan, 6.0])           # even number of sold items: (2 + 6) / 2
    floor = pv_floor(scale, unsold=np.array([False, False, True, False]),
                     items=np.array([False, True, True, True]))
    assert floor[2] == pytest.approx(4e-4)


def test_pv_floor_without_missing_scales_is_unchanged():
    scale = np.array([4.0, 0.0, 2.5e6])
    np.testing.assert_array_equal(pv_floor(scale), PV_FLOOR_FACTOR * scale)
    np.testing.assert_array_equal(
        pv_floor(scale, unsold=np.zeros(3, bool), items=np.ones(3, bool)), PV_FLOOR_FACTOR * scale)


def test_pv_floor_missing_scale_needs_the_masks():
    scale, unsold, items = _m5_case()
    with pytest.raises(ValueError, match="unsold"):
        pv_floor(scale)


def test_pv_floor_raises_for_a_missing_scale_the_rule_does_not_cover():
    scale, unsold, items = _m5_case()
    sold_but_short = unsold.copy()
    sold_but_short[5] = False                              # has sold, but still has no scale
    with pytest.raises(ValueError, match="no rule"):
        pv_floor(scale, unsold=sold_but_short, items=items)
    above = unsold.copy()
    above[0], scale[0] = True, np.nan                      # the store itself with no sale
    with pytest.raises(ValueError, match="items only"):
        pv_floor(scale, unsold=above, items=items)
    with pytest.raises(ValueError, match="No item has sold"):
        pv_floor(np.array([np.nan, np.nan]), unsold=np.array([True, True]), items=np.array([True, True]))


def test_unsold_item_reconciles_with_its_zero_forecast_and_floor():
    # store = sold item + unsold item. The unsold item has a zero forecast and zero variance.
    S = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    y_hat = np.array([10.0, 8.0, 0.0])
    scale = np.array([50.0, 4.0, np.nan])
    floor = pv_floor(scale, unsold=np.array([False, False, True]), items=np.array([False, True, True]))
    v = np.array([9.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="zero"):
        W_pv(variance=v)
    W = W_pv(variance=v, floor=floor)
    np.testing.assert_allclose(np.diag(W), [9.0, 1.0, 4e-4])
    out = reconcile(y_hat, S, W)
    assert out[0] == pytest.approx(out[1] + out[2])
    assert abs(out[2]) < 1e-3                              # the unsold item stays at about zero
    # the gap of 2 is split between the store and the sold item in proportion to their variances
    assert out[1] == pytest.approx(8.0 + 2.0 * 1.0 / (9.0 + 1.0 + 4e-4), abs=1e-9)
