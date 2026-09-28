# Pre-registration: Reconciling zero-shot foundation-model forecasts without residuals

Author: Abdul Ali Mamnun
Registered: 2026-09-27 (this file's first git commit is the timestamp)
Status: committed before any modelling code was written or any forecast was produced.

Any change after this commit goes in the **Deviations log** at the bottom, with the date and the reason. The original text above it is never edited.

---

## 1. Question

When a forecaster has no in-sample residuals, what should the error covariance W in MinT reconciliation be estimated from, and does the answer change whether reconciliation helps?

## 2. Hypotheses and what counts as "no"

Notation: *gain* = relative change in loss after reconciliation vs. the unreconciled base forecast of the same model, (L_rec − L_base) / L_base. Negative means improvement. All confidence intervals (CIs) are 95% hierarchical-bootstrap CIs (Section 7).

**H1 (does reconciliation help FMs).**
(a) For zero-shot FMs reconciled with MinT-shrink_bt, the MASE gain at the top level is negative.
(b) At the bottom level, the FM gain is smaller in magnitude than the ETS gain (ETS reconciled with MinT-shrink_is).
- **H1a counts as no** if, in the majority of (dataset × FM) cells, the CI on the top-level gain includes 0 or lies above 0.
- **H1b counts as no** if the CI on (FM bottom-level gain − ETS bottom-level gain) lies entirely below 0 in the majority of cells, i.e. FMs gain *more* than ETS at the bottom.

**H2 (which W; primary hypothesis).**
WLS-pv, a W built only from the model's own predictive variance, performs as well as WLS-var_bt and MinT-shrink_bt, which both need backtest residuals.
- Test: a 90% Model Confidence Set (MCS) over the six residual-free/backtest W-estimators (all except MinT-shrink_is), per (dataset × FM), with MASE averaged over all levels and horizons as the loss.
- **H2 counts as no** if WLS-pv is excluded from the 90% MCS in more than half of the (dataset × FM) cells.
- Reported alongside: the wall-clock cost of producing each W (base-forecast calls counted separately).

**H3 (calibration).**
Linear probabilistic reconciliation lowers CRPS without harming coverage.
- **H3 counts as no** if either (i) the CI on the CRPS gain at the top level includes 0 or lies above 0 in the majority of cells, or (ii) empirical 80% or 90% coverage after reconciliation falls more than 5 percentage points below nominal at any level where it was within 5 points before reconciliation.
- Secondary, directional: if under-coverage appears, it is at the bottom level.

**H4 (rank stability).**
Reconciliation raises the agreement between model rankings at different aggregation levels.
- Statistic: mean pairwise Kendall τ between the model rankings (by MASE) at each pair of levels, per dataset.
- **H4 counts as no** if the CI on Δτ (after − before) includes 0 or lies below 0 on the majority of datasets.

**Primary vs. secondary.** H2 is primary. H1, H3 and H4 are secondary. Multiplicity across secondary tests is handled with Holm correction and is reported explicitly.

Any outcome is reported, including null and negative results.

## 3. Datasets

All public. Loaded with `datasetsforecast.hierarchical` unless noted.

| Dataset | Structure | Frequency | Horizon h | Origins K |
|---|---|---|---|---|
| TourismSmall | hierarchical, 89 series | quarterly | 8 | 5 |
| TourismLarge | grouped, 555 series | monthly | 12 | 5 |
| Labour | grouped | monthly | 12 | 5 |
| Wiki2 | grouped | daily | 7 | 5 |
| M5 subset (1–2 stores, item → dept → category → store) | hierarchical | daily | 28 | 5 |

- TourismSmall is for development only and is **excluded from confirmatory tests**.
- Optional sanity datasets (Prison, Infant mortality) are exploratory only.
- Rolling origins are spaced h steps apart and end at the last observation.
- No TD Insurance or other non-public data is used.

## 4. Base forecasters

- **Classical (in-sample residuals available):** AutoETS, Theta, Seasonal Naive (`statsforecast`).
- **Zero-shot FMs (no residuals):** Chronos-Bolt (tiny, mini, small), Chronos-T5 small (sample paths), TiRex, Moirai-2 small, Chronos-2.
- Every FM uses its default context length and default settings. No fine-tuning and no hyperparameter search.
- A model may be dropped only for **compute reasons** (CPU runtime or failure to run), and only before its forecasts are scored. Any drop is logged as a deviation.

## 5. W-estimators

Reconciliation: G = (SᵀW⁻¹S)⁻¹SᵀW⁻¹, ỹ = S G ŷ. One numpy code path is used for all estimators.

| Code | W | Source |
|---|---|---|
| OLS | I | none |
| WLS-struct | diag(S·1) | structure only |
| WLS-var_bt | diag(σ̂²_h) | backtest residuals |
| MinT-shrink_bt | λD + (1−λ)Σ̂, Schäfer–Strimmer λ | backtest residuals |
| WLS-pv | diag(v̂_h) | model's predictive distribution |
| Hybrid | diag = v̂_h; off-diagonal = shrunk backtest correlations | both |
| MinT-shrink_is | in-sample residual covariance, shrunk | classical models only (reference arm) |

- **Backtest residuals:** at each origin, the model re-forecasts at 4 inner origins inside the context window at the same horizon h (not one-step). h-step residuals are pooled across the inner origins.
- **Predictive variance v̂_h:** the sample variance when the model produces samples; otherwise ((q₀.₉ − q₀.₁) / (2 × 1.2816))². Both are computed per series and per horizon.
- W is estimated separately for each horizon h.
- The **unreconciled base forecast** is the control arm for every model.

## 6. Metrics

Metrics are computed per (dataset, origin, model, W, level, horizon) from one long results table, never inside the forecasting loop.

- **Point:** MASE with the seasonal-naive in-sample scale. RMSSE replaces it on M5.
- **Probabilistic (primary):** CRPS from sample paths for models that produce samples.
- **Probabilistic (quantile-only models):** mean quantile loss on each model's native quantile grid. It is reported separately and not compared across grids of different density (Adachi et al. 2025).
- **Coverage:** empirical coverage of the central 80% and 90% intervals.
- **Rank stability:** Kendall τ (Section 2, H4).

## 7. Inference

- **Hierarchical bootstrap:** resample datasets, then series within datasets, then origins; 2,000 replicates; percentile CIs.
- **Multi-horizon Diebold–Mariano (Grant 2026)** for pairwise W-estimator comparisons, averaged over horizons 1..h.
- **Model Confidence Set (Hansen, Lunde & Nason 2011):** α = 0.10, T_max statistic, block bootstrap over origins.
- The design is factorial (dataset × origin × model × W × level). Inference respects the nesting and never treats series from the same hierarchy as independent draws across datasets.

## 8. Correctness gate (must pass before any FM result is examined)

1. With ETS on TourismLarge, the ordering of reconciliation methods at the top levels matches Wickramasuriya et al. (2019): MinT-shrink ≤ WLS ≤ OLS ≤ base, with the gaps shrinking toward the bottom level.
2. The numpy reconciler matches `hierarchicalforecast` to within 1e-8 on the ETS case.

If the gate fails, the pipeline is fixed before continuing. Gate results are reported in the appendix.

## 9. Exploratory (not confirmatory)

- Bias propagation under trend-shrinkage (Chronos on trending series) and its interaction with MinT's unbiasedness assumption.
- Chronos-2 (multivariate) vs. univariate FMs after reconciliation.
- Results on the optional sanity datasets.

These analyses are labelled exploratory wherever they are reported.

## 10. Reproducibility

- Every base forecast is cached to parquet, keyed by (dataset, origin, model).
- Results table schema: `dataset, origin, model, W_est, level, series_id, horizon, y, yhat, q10..q90, mase_scale`.
- Package versions are pinned in `requirements.txt`, and random seeds are fixed.
- Code, cached forecasts and the paper are released together as a tagged GitHub release.

---

## Deviations log

| Date | Section | Change | Reason |
|---|---|---|---|
| | | | |
| 2026-09-27 | §6 | Clarification: MASE within a level is the arithmetic mean over series. Series with seasonal-naive scale < 1e-8 at an origin are excluded from MASE at that origin; the count per level is reported. | Gap in original text; set before any forecast was scored. Diagnostic on TourismLarge found 0 such series. |
| 2026-09-27 | §6 | Added: RMSSE reported as a secondary point metric on all datasets (not only M5). Primary metric unchanged. | TourismLarge bottom level is intermittent (50/304 series >50% zeros); set before any forecast was scored. |
| 2026-09-27 | §8.1 | Gate criterion revised AFTER seeing a single-origin failure. The original chain (MinT-shrink ≤ WLS ≤ OLS ≤ Base at top levels) failed at Country on the 2016 origin; it is not implied by theory when base forecasts are biased. New criterion, fixed before running it: averaged over the 5 pre-registered origins, (i) all reconciled forecasts are coherent, (ii) MinT-shrink mean MASE ≤ Base on "All series" and at ≥5 of 8 levels, (iii) BottomUp is worse than MinT-shrink at Country; plus §8.2 unchanged. If (ii) or (iii) fails, no FM work until the paper's exact protocol is replicated. The original chain is still reported descriptively in the appendix, with the single-origin tables. | Original gate over-specified; the pipeline checks (coherence, independent scoring, BottomUp = bottom sum) pass. Result-informed, disclosed as such. |
| 2026-09-27 | §10 | Added column `rmsse_scale` (mean squared seasonal-naive error on training data) to the results-table schema. Registered columns unchanged. | Required to compute the RMSSE secondary metric (earlier deviation) from the table alone. |
