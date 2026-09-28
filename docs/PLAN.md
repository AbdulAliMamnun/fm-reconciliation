# Research plan: Reconciling zero-shot foundation-model forecasts without residuals

Author: Abdul Ali Mamnun (solo). Compute: MacBook (Apple silicon / CPU), $0 budget. Timeline: ~12 weeks from start.
Prepared 2026-09-27. Self-contained: everything needed to start in a fresh session is here.

---

## 1. The research question (one sentence)

**When a forecaster has no in-sample residuals, what should the error covariance W in MinT reconciliation be estimated from, and does the answer change whether reconciliation helps?**

Working title options:
- "Reconciling foundation-model forecasts without residuals: predictive variance as a covariance estimate for MinT"
- "Does hierarchical reconciliation help zero-shot time-series foundation models?"

## 2. Why this is open

- Hierarchical reconciliation (MinT, Wickramasuriya et al. 2019) reliably improves ETS/ARIMA base forecasts, but every good estimator of W is built from in-sample residuals.
- Zero-shot foundation models (Chronos-Bolt, TiRex, Moirai-2, Chronos-2, TimesFM) forecast every series independently from a context window and produce **no residuals**. The standard MinT recipe cannot be run as written.
- The only prior work is a Nixtla TimeGPT tutorial (MinTrace-OLS and mint_shrink on Australian tourism, single dataset, RMSE only, no treatment of where residuals come from, nothing on calibration). No systematic study exists as of Sept 2026 (searched arXiv, Google Scholar, Nixtla docs).
- Foundation models give something ETS does not: a predictive distribution per series per horizon, i.e. a residual-free estimate of the diagonal of W at the right horizon. Nobody has tested whether that is a sufficient W.
- Related evidence that makes the question matter: FM rankings flip across aggregation levels (Islam & Mohammed, arXiv 2609.27867, Sept 2026; TIME benchmark authors call aggregation level an open question); FMs are roughly calibrated (~5% calibration error, Adler et al. ICLR 2026); Chronos wins by shrinking trend, i.e. is biased on trending series (Cherif, arXiv 2607.19383) and MinT theory assumes unbiased base forecasts.

## 3. Hypotheses (pre-register these)

- **H1 (does it help):** Reconciliation improves zero-shot FM point accuracy at aggregate levels; gain at the bottom level is smaller than for ETS.
- **H2 (which W, the core):** W built from the model's own predictive variance (WLS-pv) is statistically indistinguishable from W built from backtest residuals (MinT-shrink_bt) under a Model Confidence Set, at a fraction of the compute.
- **H3 (calibration):** Linear probabilistic reconciliation improves CRPS without destroying coverage; if under-coverage appears it is at the bottom level.
- **H4 (rank stability):** Reconciliation raises Kendall τ between model rankings computed at different aggregation levels.

Any outcome of H1/H2/H3 is publishable if the inference is careful. The failure mode is noisy, inconsistent results across datasets; the hierarchical bootstrap reports that honestly.

## 4. Design

### Datasets (all public, loaders in `datasetsforecast.hierarchical`)
- Australian Tourism: `TourismSmall` (89 series, for development) then `TourismLarge` (555 series, 4 levels; the MinT-paper dataset). Horizon 8 quarters / 12 months.
- Australian Labour force (grouped).
- Wikipedia page views (grouped hierarchy, `Wiki2`).
- M5 subset: 1–2 stores, item → category → department → store. Horizon 28 days. Use RMSSE not MASE (intermittent series).
- Optional sanity: Prison population, Infant mortality (fpp3).

### Base forecasters
- Classical (have residuals): ETS (`statsforecast.AutoETS`), Theta, Seasonal Naive.
- Zero-shot FMs (no residuals): Chronos-Bolt tiny/mini/small (`chronos-forecasting`), Chronos-T5 small (samples, for the sample-path arm), TiRex (`tirex`, check CPU support), Moirai-2 small (`uni2ts`), Chronos-2 (multivariate; sees related series jointly).

### The seven W-estimators (one numpy code path)
| Code | W | Needs | Status |
|---|---|---|---|
| OLS | I | nothing | baseline |
| WLS-struct | diag(S·1) | S only | baseline |
| WLS-var_bt | diag(σ̂²) from backtest residuals | re-forecast inside context, h-step | main comparator |
| MinT-shrink_bt | λD + (1−λ)Σ̂ from backtest residuals (Schäfer–Strimmer λ) | same | main comparator |
| **WLS-pv** | diag(predictive variance at h) from the model's quantiles | nothing extra | **NEW** |
| **Hybrid** | diag = predictive variance; off-diag = shrunk backtest correlations | both | **NEW** |
| MinT-shrink_is | in-sample residual covariance | classical models only | ETS reference arm |

Reconciliation: G = (SᵀW⁻¹S)⁻¹SᵀW⁻¹, ỹ = S·G·ŷ. Probabilistic: apply S·G to sample paths (or quantiles via linear probabilistic reconciliation, Panagiotelis et al. 2023).

### Protocol
- Rolling origin, K = 5–10 origins per dataset, horizon matched to dataset.
- Backtest residuals: re-forecast at n_windows origins inside the context window, at horizon h (not one-step).
- Cache every base forecast to disk keyed by (dataset, origin, model). Reconciliation is cheap; forecasting is not.
- Unreconciled forecasts are the control arm for every model.

### Metrics (per level, horizon, origin)
- Point: MASE (RMSSE on M5).
- Probabilistic: CRPS computed sample-based or on a dense quantile grid (NOT the 9-quantile weighted quantile loss; Adachi et al. 2025 show it is biased), empirical coverage at 80% and 90%.
- Rank stability: Kendall τ of model rankings across aggregation levels, before vs after reconciliation.

### Inference
- Hierarchical bootstrap: series nested within dataset, for CIs on gain vs unreconciled.
- Multi-horizon Diebold–Mariano (Grant 2026, J. Forecasting) between W-estimators.
- Model Confidence Set (Hansen et al.) over W-estimators.
- The design is factorial (dataset × origin × model × W × level); inference must respect the nesting.

### Results table schema (one long table, everything appends to it)
`dataset, origin, model, W_est, level, series_id, horizon, y, yhat, q10, q20, ..., q90, mase_scale`
Metrics are computed from this table afterwards, never inside the loop.

## 5. Week-by-week plan

| Week | Goal | Done when |
|---|---|---|
| 1 | Correctness gate (classical only) | ETS + MinT on TourismLarge reproduces the ordering in Wickramasuriya et al. 2019; own numpy reconciler matches `hierarchicalforecast` to 1e-8; rolling-origin loop and results table exist |
| 2 | First FM | Chronos-Bolt-small on Tourism, all 7 W-estimators, 1 origin; first look at H1/H2 |
| 3 | Residual sources | `backtest_residuals`, `predictive_variance`, `insample_residuals` with one signature; Hybrid W working |
| 4–5 | Scale out models | TiRex, Moirai-2, Chronos-T5, Chronos-2 on Tourism; drop any model whose CPU path is a time sink |
| 5–6 | Scale out datasets | Labour, Wiki, M5 subset; all forecasts cached |
| 6 | Probabilistic arm | S·G applied to sample paths / quantiles; CRPS and coverage columns filled |
| 7–8 | Inference + figures | hierarchical bootstrap, multi-horizon DM, MCS, Kendall τ; 4 figures |
| 9–11 | Write | 4–9 page workshop draft; optional theory appendix on bias propagation under trend-shrinkage |
| 12 | Buffer / submit | tag a GitHub release with the draft PDF; arXiv; workshop submission |

Four figures: (1) gain vs unreconciled by level, per model; (2) W-estimator comparison with bootstrap CIs; (3) coverage before/after reconciliation; (4) rank stability across levels.

## 6. Week 1, day by day

**Day 1: repo + pre-registration.**
- Create GitHub repo `fm-reconciliation`.
- Write `PREREG.md`: the question, H1–H4, datasets, models, the seven W-estimators, metrics, inference plan, and what result counts as "no". Commit it before any code.

**Day 1–2: environment.**
```
python3.11 -m venv .venv && source .venv/bin/activate
pip install hierarchicalforecast statsforecast datasetsforecast utilsforecast pandas numpy scipy matplotlib
```
- Load `datasetsforecast.hierarchical.HierarchicalData.load('./data', 'TourismSmall')`; confirm `Y_df`, `S_df`, `tags`. Then `TourismLarge`.

**Day 2–4: reproduce MinT with ETS.**
- `AutoETS` on all series, holdout 8 quarters.
- Reconcile with `BottomUp`, `MinTrace(method='ols')`, `MinTrace(method='wls_struct')`, `MinTrace(method='wls_var')`, `MinTrace(method='mint_shrink')`.
- MASE per level. Required pattern: MinT-shrink ≤ WLS ≤ OLS ≤ base at top levels, gaps shrinking toward the bottom. If BottomUp beats MinT-shrink at the total, the pipeline is wrong. Exact numbers will differ from the paper (different ETS implementation); the ordering must not.

**Day 4–6: own reconciler + tests.**
- `reconcile(y_hat, S, W) -> y_tilde` in numpy.
- Builders: `W_ols`, `W_struct(S)`, `W_var(resid)`, `W_shrink(resid)`, and stubs for `W_pv(quantiles)`, `W_hybrid(quantiles, resid)`.
- `pytest`: assert equality with `hierarchicalforecast` to 1e-8 on the ETS case.

**Day 6–8: rolling origins + results table.**
- Loop over 5 origins; write every forecast to the long-format table above (parquet).
- Metric functions read from the table.

End of week 1: verified pipeline, pre-registration, results table, no FM touched yet.

## 7. Traps
- `hierarchicalforecast` needs `S_df` and `tags` in its exact format; get this right on Tourism first.
- `MinTrace` expects an in-sample fitted-values frame; for FMs either feed backtest residuals through that interface or (better) use the numpy reconciler for all seven W's.
- M5 intermittent series make MASE degenerate; use RMSSE.
- TiRex CPU support is not guaranteed; check the README before depending on it.
- Chronos-Bolt outputs quantiles, not samples; keep Chronos-T5 small for the sample-path arm.
- Anything using TD data needs employer clearance; use public data only.

## 8. Reading list (read closely, in this order)
1. Wickramasuriya, Athanasopoulos & Hyndman (2019), JASA, "Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization." (MinT; Table 5 tourism results to reproduce.)
2. Panagiotelis, Gamakumara, Athanasopoulos & Hyndman (2023), EJOR, "Probabilistic forecast reconciliation: properties, evaluation and score optimisation."
3. Ansari et al. (2025), "Chronos-2", arXiv 2510.15821, evaluation section.
4. Skim: Hyndman & Athanasopoulos, FPP3 chapter 11 (plain-English MinT). Adachi et al. 2025, arXiv 2503.06079 (CRPS estimator bias). Adler et al. ICLR 2026, arXiv 2510.16060 (FM calibration). Islam & Mohammed 2026, arXiv 2609.27867 (aggregation level flips rankings). Nixtla TimeGPT hierarchical tutorial (the prior demo to position against).

## 9. Venues and admin (handle later, not now)
- ICLR 2027 workshops: list posted ~Nov 29, 2026; deadlines ~Feb 1, 2027; 4–9 pages; mostly non-archival (can extend later).
- ISF 2027 abstract (~March 2027): the forecasting community.
- Extended version: ECML-PKDD 2027 (~March), IJF, or TMLR (rolling).
- arXiv: first-time unaffiliated authors need an endorsement (of the author, not the paper) from an existing arXiv author in stat.ML/cs.LG. Ask once there is a draft and a repo with results (week 8–9); attach the GitHub release. Not needed to submit to workshops or TMLR.
- OpenReview profile: create early; non-institutional emails can take ~2 weeks to approve.
- Priority timestamp before arXiv: tagged GitHub release with the draft PDF (Zenodo will mint a DOI for it).

## 10. Software
`hierarchicalforecast`, `statsforecast`, `datasetsforecast`, `utilsforecast` (Nixtla); `chronos-forecasting` (Amazon); `tirex` (NX-AI); `uni2ts` (Salesforce, Moirai); `torch` (MPS or CPU); `scipy`, `numpy`, `pandas`, `pyarrow`, `matplotlib`, `pytest`. Optional: `sktime` wrapper as a follow-up contribution.
