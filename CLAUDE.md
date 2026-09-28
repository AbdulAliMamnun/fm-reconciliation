# fm-reconciliation

Research code for a pre-registered study: reconciling zero-shot time-series foundation-model forecasts when there are no in-sample residuals.

Read `PREREG.md` and `docs/PLAN.md` before any work.

## Hard rules

- NEVER edit `PREREG.md` above the "Deviations log". Any design change goes in that log with the date and reason, and only after asking me first.
- Public data only. Never use TD Insurance or other employer data.
- No foundation-model code until the correctness gate (PREREG §8) passes and I've confirmed it.
- Don't push to GitHub without asking. Commit locally in small steps with clear messages.

## Environment

- Conda env `fmrec`, Python 3.11. Activate it with `conda activate fmrec` before running anything.
- Pin every new package in `requirements.txt`.
- Hardware is a MacBook (Apple silicon), CPU/MPS only, $0 budget. Keep runs cheap and cache everything.

## Code layout

- `src/`: library code (data loading, forecasters, W-estimators, reconciler, metrics, inference).
- `tests/`: pytest tests. Every non-trivial function gets a test.
- `notebooks/`: throwaway exploration scripts.
- `data/`: downloaded datasets (gitignored).
- `results/`: cached forecasts and the results table as parquet (gitignored).

## Design invariants

- Reconciliation for ALL W-estimators goes through ONE numpy function: `reconcile(y_hat, S, W) -> y_tilde`, with G = (SᵀW⁻¹S)⁻¹SᵀW⁻¹. Use `np.linalg.solve`, never an explicit inverse.
- Every base forecast is cached to disk, keyed by (dataset, origin, model). Never re-run a forecast that is already cached.
- Everything appends to one long results table with this schema: `dataset, origin, model, W_est, level, series_id, horizon, y, yhat, q10..q90, mase_scale`.
- Metrics are computed from the results table afterwards, never inside the forecast loop.
- Unreconciled base forecasts are always kept as the control arm.
- Backtest residuals are h-step (not one-step), from re-forecasting inside the context window.

## Known traps

- `hierarchicalforecast` needs `S_df` and `tags` in its exact format. Check against TourismSmall first.
- `MinTrace` expects in-sample fitted values. For FMs, use our numpy reconciler instead.
- M5 is intermittent, so use RMSSE there, not MASE.
- Chronos-Bolt and TiRex output 9 quantiles, not samples. Chronos-T5 is the sample-path arm.
- Check TiRex CPU support before depending on it.

## Working style

- Do one step at a time, then stop and show me the output.
- Run the tests before saying something works.
- If a result looks surprising, say so and investigate. Don't paper over it.
- Explain non-obvious statistical choices briefly. I need to defend them in the paper.
