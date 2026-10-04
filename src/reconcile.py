"""MinT reconciliation in numpy: one code path for every W-estimator (PREREG §5).

    G = (S' W^-1 S)^-1 S' W^-1,    y_tilde = S G y_hat

The W builders follow the conventions of hierarchicalforecast 1.5.3 so that the
correctness gate (PREREG §8.2) can compare the two to 1e-8. Source references:

    methods.py         hierarchicalforecast/methods.py, installed 1.5.3
    reconciliation.cpp src/reconciliation.cpp in the 1.5.3 sdist (compiled into
                       hierarchicalforecast/_lib)

Differences from the library that do not change the result beyond rounding:
  - The library never forms W^-1. It uses the equivalent representation
    P = J - (U'WU)^-1-solve form (methods.py:1341-1345, 1405-1412), where
    U' = [I, -S_agg] and J = [0, I]. We use the G formula above.
  - Before solving, the library sets entries of its two matrices with absolute
    value < 1e-10 to zero (methods.py:1408, 1410). We do not.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

EPS = 2e-8  # hard-coded in the library: methods.py:1376, reconciliation.cpp:112 and :201


def _solve_spd(A, B):
    """Solve A X = B for symmetric positive definite A, without inverting A."""
    try:
        return cho_solve(cho_factor(A, lower=True, check_finite=False), B, check_finite=False)
    except np.linalg.LinAlgError:
        return np.linalg.solve(A, B)


def _check(S, W):
    S = np.asarray(S, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"S must be 2-D, got shape {S.shape}.")
    n = S.shape[0]
    if W.shape != (n, n):
        raise ValueError(f"W must have shape {(n, n)}, got {W.shape}.")
    if not np.isfinite(W).all():
        raise ValueError("W contains NaN or inf.")
    return S, W


def projection_matrix(S, W):
    """G = (S' W^-1 S)^-1 S' W^-1, shape (n_bottom, n_series)."""
    S, W = _check(S, W)
    WinvS = _solve_spd(W, S)             # W^-1 S
    return _solve_spd(S.T @ WinvS, WinvS.T)


def reconcile(y_hat, S, W):
    """Reconcile base forecasts: y_tilde = S G y_hat.

    y_hat: (n_series, h), or (n_series,), in S row order.
    S: (n_series, n_bottom) summing matrix.
    W: (n_series, n_series) symmetric positive definite.
    """
    y_hat = np.asarray(y_hat, dtype=np.float64)
    S, W = _check(S, W)
    if y_hat.shape[0] != S.shape[0]:
        raise ValueError(f"y_hat has {y_hat.shape[0]} rows, S has {S.shape[0]}.")
    if not np.isfinite(y_hat).all():
        raise ValueError("y_hat contains NaN or inf.")
    return S @ (projection_matrix(S, W) @ y_hat)


def reconcile_by_horizon(y_hat, S, Ws):
    """Reconcile each horizon with its own W (PREREG §5: W is estimated
    separately for each horizon).

    y_hat: (n_series, h). Ws: sequence of h matrices, each (n_series, n_series).
    Column t of the result is reconcile(y_hat[:, t], S, Ws[t]).
    """
    y_hat = np.asarray(y_hat, dtype=np.float64)
    if y_hat.ndim != 2 or len(Ws) != y_hat.shape[1]:
        raise ValueError(f"Need one W per horizon: y_hat has shape {y_hat.shape}, got {len(Ws)} W.")
    return np.column_stack([reconcile(y_hat[:, t], S, Ws[t]) for t in range(y_hat.shape[1])])


# ---------------------------------------------------------------- W builders

def W_ols(n):
    """W = I (methods.py:1346-1348)."""
    return np.eye(n)


def W_struct(S):
    """W = diag(S 1): the number of bottom series under each series
    (methods.py:1349-1352)."""
    return np.diag(np.sum(np.asarray(S, dtype=np.float64), axis=1))


def W_var(resid):
    """W = diag(mean squared residual + 2e-8).  resid: (n_series, T).

    Conventions of hierarchicalforecast `wls_var` (methods.py:1371-1378):
      - Centering: none. It is the mean of squared residuals, not a variance.
      - Denominator: T, the full number of time points (methods.py:1374).
      - NaN: skipped in the sum (np.nansum, methods.py:1373), but T still
        counts them, so a NaN acts like a zero residual.
      - Jitter: 2e-8 is added to every diagonal entry (methods.py:1376).
    """
    resid = np.asarray(resid, dtype=np.float64)
    T = resid.shape[1]
    return np.diag(np.nansum(resid ** 2, axis=1) / T + EPS)


def W_shrink(resid, ridge=EPS, return_lambda=False):
    """Schafer-Strimmer shrinkage of the residual covariance toward its diagonal.

        W = lambda * D + (1 - lambda) * Sigma_hat,   D = diag(Sigma_hat)

    resid: (n_series, T). With return_lambda=True, returns (W, lambda).

    `lambda` is the weight on the diagonal target, as in PREREG §5. The library
    variable `shrinkage` (reconciliation.cpp:166) is 1 - lambda.

    Conventions of hierarchicalforecast `mint_shrink` when there are no NaNs
    (reconciliation.cpp:104-181, called from methods.py:1388-1401):
      - Centering: each series has its mean over T subtracted
        (reconciliation.cpp:119-123, 144).
      - Sigma_hat: denominator T-1 (reconciliation.cpp:113, 149).
      - Standardising for the correlations: population standard deviation,
        denominator T, plus 2e-8 (reconciliation.cpp:130).
      - lambda, with w_k = xs_ik * xs_jk the product of standardised residuals
        and wbar its mean over T:
            lambda = clip( [sum_{i>j} sum_k (w_k - wbar)^2 / (T (T-1))]
                           / [sum_{i>j} wbar^2 + 2e-8], 0, 1 )
        (reconciliation.cpp:114-115, 153-160, 166-169). The sums run over the
        lower triangle only, and 2e-8 is added to the denominator.
      - Clipping: lambda is clipped to [0, 1] (std::clamp, reconciliation.cpp:167).
      - Diagonal: not shrunk. It is floored at `ridge` (default 2e-8, the
        library's mint_shr_ridge): W_ii = max(Sigma_hat_ii, ridge)
        (reconciliation.cpp:178). This is a floor, not an added jitter.

    With any NaN the library switches to a pairwise-complete version
    (methods.py:1390-1395, reconciliation.cpp:189-312), reproduced in
    `_W_shrink_nan`. It is not the same formula with NaNs dropped: see there.
    """
    resid = np.asarray(resid, dtype=np.float64)
    if resid.ndim != 2 or resid.shape[1] < 2:
        raise ValueError(f"resid must have shape (n_series, T) with T >= 2, got {resid.shape}.")
    if np.isnan(resid).any():
        W, lam = _W_shrink_nan(resid, ridge)
        return (W, lam) if return_lambda else W

    n, T = resid.shape
    X = resid - resid.mean(axis=1, keepdims=True)
    cross = X @ X.T                                   # sum_k c_k, with c_k = x_ik x_jk
    cross_sq = (X ** 2) @ (X ** 2).T                  # sum_k c_k^2
    inv_std = 1.0 / (np.sqrt(np.einsum("ik,ik->i", X, X) / T) + EPS)
    s = np.outer(inv_std, inv_std)

    cbar = cross / T
    var_w = s ** 2 * (cross_sq - T * cbar ** 2)       # sum_k (w_k - wbar)^2
    sq_corr = (s * cbar) ** 2                         # wbar^2
    low = np.tril_indices(n, k=-1)
    lam = (var_w[low].sum() / (T * (T - 1))) / (sq_corr[low].sum() + EPS)
    lam = float(np.clip(lam, 0.0, 1.0))

    cov = cross / (T - 1)
    W = (1.0 - lam) * cov
    W[np.diag_indices(n)] = np.maximum(np.diag(cov), ridge)
    return (W, lam) if return_lambda else W


def _W_shrink_nan(resid, ridge):
    """Pairwise-complete shrinkage, as in reconciliation.cpp:189-312.

    For each pair (i, j) only the time points where both series are observed
    are used, with count n_ij:
      - Means and centering are per pair (reconciliation.cpp:218-242).
      - Sigma_hat_ij has denominator n_ij - 1 (reconciliation.cpp:243-244).
        Pairs with n_ij <= 1 are left at 0 (reconciliation.cpp:217).
      - The standardising divisor is the population standard deviation plus
        2e-8, plus another 2e-8: the library adds it twice
        (reconciliation.cpp:259-260, then 266-267).
      - Numerator term: n_ij / (n_ij - 1)^3 * sum_k (w_k - wbar)^2
        (reconciliation.cpp:248-250, 288). Denominator term:
        (n_ij / (n_ij - 1) * wbar)^2 (reconciliation.cpp:290-291).
      - lambda = clip(numerator / (denominator + 2e-8), 0, 1)
        (reconciliation.cpp:298-300); the diagonal is floored at `ridge`
        (reconciliation.cpp:309).
    """
    n, _ = resid.shape
    mask = ~np.isnan(resid)
    R = np.where(mask, resid, 0.0)
    W = np.zeros((n, n))
    num = den = 0.0

    for i in range(n):
        m = mask[: i + 1] & mask[i]                   # joint mask with every j <= i
        count = m.sum(axis=1).astype(np.float64)
        ok = count > 1
        if not ok.any():
            continue
        m, count, J = m[ok], count[ok][:, None], np.flatnonzero(ok)

        ri = np.where(m, R[i], 0.0)
        rj = np.where(m, R[J], 0.0)
        xi = np.where(m, ri - ri.sum(axis=1, keepdims=True) / count, 0.0)
        xj = np.where(m, rj - rj.sum(axis=1, keepdims=True) / count, 0.0)
        W[i, J] = W[J, i] = (xi * xj).sum(axis=1) / (count[:, 0] - 1)

        off = J != i
        if not off.any():
            continue
        m, count, xi, xj = m[off], count[off], xi[off], xj[off]
        std_i = np.sqrt((xi ** 2).sum(axis=1, keepdims=True) / count) + EPS
        std_j = np.sqrt((xj ** 2).sum(axis=1, keepdims=True) / count) + EPS
        xs_i = xi / (std_i + EPS)
        xs_j = xj / (std_j + EPS)
        xs_i = np.where(m, xs_i - xs_i.sum(axis=1, keepdims=True) / count, 0.0)
        xs_j = np.where(m, xs_j - xs_j.sum(axis=1, keepdims=True) / count, 0.0)
        w = xs_i * xs_j
        wbar = w.sum(axis=1, keepdims=True) / count
        var_w = np.where(m, (w - wbar) ** 2, 0.0).sum(axis=1)
        c = count[:, 0]
        num += float((c / (c - 1) ** 3 * var_w).sum())
        den += float(((c / (c - 1) * wbar[:, 0]) ** 2).sum())

    lam = float(np.clip(num / (den + EPS), 0.0, 1.0))
    diag = np.diag(W).copy()
    W *= 1.0 - lam
    W[np.diag_indices(n)] = np.maximum(diag, ridge)
    return W, lam


Z90 = 1.2816  # standard normal 0.9 quantile, to the 4 decimals registered in PREREG §5


def predictive_variance(quantiles):
    """v_hat = ((q90 - q10) / (2 * 1.2816))^2 (PREREG §5).

    quantiles: (..., 9), the last axis holding q10, q20, ..., q90. Only q10 and
    q90 are used. For a normal distribution the 10%-90% range is 2 * 1.2816
    standard deviations, so v_hat is the variance of the normal that has the
    same 80% interval as the model.
    """
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.shape[-1] != 9:
        raise ValueError(f"Expected 9 quantiles (q10..q90) on the last axis, got {quantiles.shape}.")
    if not np.isfinite(quantiles).all():
        raise ValueError("quantiles contains NaN or inf.")
    return ((quantiles[..., -1] - quantiles[..., 0]) / (2.0 * Z90)) ** 2


PV_FLOOR_FACTOR = 1e-4   # Deviations log: v_hat = max(v_hat, 1e-4 * rmsse_scale) per series


def pv_floor(rmsse_scale, unsold=None, items=None):
    """The floor of the predictive variance, per series (Deviations log, §5).

    rmsse_scale is the mean squared seasonal-naive error on the training data,
    so the floor is in the units of a variance. It says that no forecast is
    trusted more than 100 times (in standard deviation) the seasonal-naive
    forecast of the same series.

    M5 rule (Deviations log, 2026-10-04): an item with no sale before the origin
    has no rmsse_scale. Its floor uses the median rmsse_scale of the items that
    have sold at that origin.
        unsold: bool (n_series,), True for a series with no sale before the origin.
        items:  bool (n_series,), True for the bottom-level series.
    Both are needed when any scale is missing. A scale that is missing for any
    other reason is not covered by the registration and raises.
    """
    scale = np.asarray(rmsse_scale, dtype=np.float64).copy()
    missing = ~np.isfinite(scale)
    if not missing.any():
        return PV_FLOOR_FACTOR * scale
    if unsold is None or items is None:
        raise ValueError(f"{int(missing.sum())} series have no rmsse_scale; pass `unsold` and `items`.")
    unsold, items = np.asarray(unsold, dtype=bool), np.asarray(items, dtype=bool)
    if unsold.shape != scale.shape or items.shape != scale.shape:
        raise ValueError("unsold and items must have the shape of rmsse_scale.")
    if (unsold & ~items).any():
        raise ValueError(f"{int((unsold & ~items).sum())} series above the item level have no sale "
                         "before the origin; the registered rule covers items only.")
    other = missing & ~unsold
    if other.any():
        raise ValueError(
            f"{int(other.sum())} series have sold but have no rmsse_scale (too few observations "
            "since the first sale). The Deviations log has no rule for them."
        )
    sold = items & ~unsold & ~missing
    if not sold.any():
        raise ValueError("No item has sold before the origin, so there is no median scale.")
    scale[unsold] = np.median(scale[sold])
    return PV_FLOOR_FACTOR * scale


def _floored(v, floor):
    """Apply the per-series floor to variances of shape (n_series,) or (n_series, h)."""
    v = np.asarray(v, dtype=np.float64)
    if not np.isfinite(v).all() or (v < 0).any():
        raise ValueError("The predictive variance contains NaN, inf or negative values.")
    if floor is not None:
        floor = np.asarray(floor, dtype=np.float64)
        if floor.shape != v.shape[:1]:
            raise ValueError(f"floor must have shape ({v.shape[0]},), got {floor.shape}.")
        v = np.maximum(v, floor.reshape((-1,) + (1,) * (v.ndim - 1)))
    if (v <= 0).any():
        raise ValueError(
            f"{int((v <= 0).sum())} predictive variances are zero; W would not be positive "
            "definite. Pass the floor of the Deviations log (pv_floor)."
        )
    return v


def W_pv(quantiles=None, *, variance=None, floor=None):
    """W = diag(v_hat) (PREREG §5, WLS-pv).

    quantiles: (n_series, 9) for one horizon, columns q10..q90 in S row order;
        v_hat is then `predictive_variance(quantiles)`.
    variance: v_hat given directly, (n_series,). For sample-path models it is
        the sample variance (PREREG §5). Give either quantiles or variance.
    floor: per-series floor (n_series,), see `pv_floor`. Without it, a zero
        variance raises.

    For every horizon at once, pass quantiles (n_series, h, 9) or variance
    (n_series, h) and get a list of h matrices for `reconcile_by_horizon`.
    """
    if (quantiles is None) == (variance is None):
        raise ValueError("Give either quantiles or variance.")
    if quantiles is not None:
        quantiles = np.asarray(quantiles, dtype=np.float64)
        if quantiles.ndim not in (2, 3):
            raise ValueError(
                f"quantiles must be (n_series, 9) or (n_series, h, 9), got {quantiles.shape}.")
        variance = predictive_variance(quantiles)
        if floor is None and (quantiles[..., -1] <= quantiles[..., 0]).any():
            n_bad = int((quantiles[..., -1] <= quantiles[..., 0]).sum())
            raise ValueError(f"{n_bad} series have q90 <= q10; W_pv would not be positive definite.")
    v = _floored(variance, floor)
    if v.ndim == 2:
        return [np.diag(v[:, t]) for t in range(v.shape[1])]
    if v.ndim != 1:
        raise ValueError(f"variance must be (n_series,) or (n_series, h), got {v.shape}.")
    return np.diag(v)


def _per_horizon(resid):
    """Backtest residuals (n_series, n_inner, h) as a list of h arrays (n_series, n_inner)."""
    resid = np.asarray(resid, dtype=np.float64)
    if resid.ndim != 3:
        raise ValueError(f"resid must have shape (n_series, n_inner, h), got {resid.shape}.")
    return [resid[:, :, t] for t in range(resid.shape[2])]


def W_var_bt(resid):
    """WLS-var_bt: one W_var per horizon, from that horizon's backtest residuals
    (PREREG §5 and Deviations log). It is the uncentered mean squared h-step
    error, so a biased series gets a larger variance.

    resid: (n_series, n_inner, h). Returns a list of h diagonal matrices.
    """
    return [W_var(r) for r in _per_horizon(resid)]


def W_shrink_bt(resid, ridge=EPS, return_lambda=False):
    """MinT-shrink_bt: one W_shrink per horizon, from that horizon's backtest
    residuals. Centered, denominator T-1, lambda clipped to [0, 1], as in W_shrink.

    resid: (n_series, n_inner, h). Returns a list of h matrices, and with
    return_lambda=True also the list of h shrinkage intensities.
    """
    out = [W_shrink(r, ridge=ridge, return_lambda=True) for r in _per_horizon(resid)]
    Ws, lams = [w for w, _ in out], [lam for _, lam in out]
    return (Ws, lams) if return_lambda else Ws


def W_hybrid(quantiles=None, resid=None, return_lambda=False, *, variance=None, floor=None):
    """Hybrid (PREREG §5): variances from the model's predictive distribution,
    correlations from the backtest residuals, shrunk toward no correlation.

        W = D_pv^1/2 R_shr D_pv^1/2
        D_pv  = diag(v_hat), as in `W_pv` (quantiles or variance, with the floor)
        R_shr = lambda I + (1 - lambda) R_hat

    R_hat is the correlation matrix of the backtest residuals and lambda is the
    shrinkage intensity that `W_shrink` computes from the same residuals. R_shr
    is therefore the correlation matrix of the W_shrink covariance, which is how
    it is computed here. A series whose residuals do not vary has correlation 0
    with every other series.

    One horizon: quantiles (n_series, 9) or variance (n_series,), resid (n_series, T).
    Every horizon: quantiles (n_series, h, 9) or variance (n_series, h), resid
    (n_series, n_inner, h); returns a list of h matrices.

    W is positive definite when lambda > 0. With lambda = 0 and fewer residuals
    than series, R_hat is singular and so is W.
    """
    resid = np.asarray(resid, dtype=np.float64)
    D = W_pv(quantiles, variance=variance, floor=floor)
    if isinstance(D, list):
        rs = _per_horizon(resid)
        if len(D) != len(rs):
            raise ValueError("The predictive variance and resid have a different number of horizons.")
        out = [_hybrid(np.diag(D[t]), rs[t]) for t in range(len(rs))]
        Ws, lams = [w for w, _ in out], [lam for _, lam in out]
        return (Ws, lams) if return_lambda else Ws
    W, lam = _hybrid(np.diag(D), resid)
    return (W, lam) if return_lambda else W


def _hybrid(v, resid):
    d_pv = np.sqrt(v)
    if resid.ndim != 2 or resid.shape[0] != d_pv.shape[0]:
        raise ValueError(f"resid must have shape ({d_pv.shape[0]}, T), got {resid.shape}.")
    W_s, lam = W_shrink(resid, return_lambda=True)
    d_s = np.sqrt(np.diag(W_s))
    R = W_s / np.outer(d_s, d_s)
    R[np.diag_indices_from(R)] = 1.0
    return d_pv[:, None] * R * d_pv[None, :], lam
