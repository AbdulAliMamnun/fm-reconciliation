#!/bin/bash
# The full confirmatory plan, forecasting and caching only, in the registered order.
# Start detached from the repo root:
#   nohup bash scripts/run_plan.sh > results/logs/plan.log 2>&1 &
# Progress: tail -f results/logs/plan.log ; cat results/logs/forecasts_status.json
# Safe to stop and start again: everything already cached is skipped.
set -u
cd "$(dirname "$0")/.."
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate fmrec
mkdir -p results/logs
ALL="chronos_bolt_tiny chronos_bolt_mini chronos_bolt_small moirai_2_small chronos_2 chronos_t5_small tirex"
echo "$(date '+%F %T')  PLAN START (pid $$)"
echo "$(date '+%F %T')  (a) TourismLarge";  python -u scripts/run_forecasts.py TourismLarge $ALL
echo "$(date '+%F %T')  (b) Labour";        python -u scripts/run_forecasts.py Labour $ALL
echo "$(date '+%F %T')  (c) M5";            python -u scripts/run_forecasts.py M5 chronos_bolt_tiny chronos_bolt_mini chronos_bolt_small moirai_2_small chronos_2 chronos_t5_small tirex
echo "$(date '+%F %T')  PLAN END"
