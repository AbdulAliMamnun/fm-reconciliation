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
    W_hybrid, W_ols, W_pv, W_shrink, W_struct, W_var, projection_matrix, reconcile,
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


def test_stubs_raise():
    with pytest.raises(NotImplementedError):
        W_pv(None)
    with pytest.raises(NotImplementedError):
        W_hybrid(None, None)
