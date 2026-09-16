#!/usr/bin/env bash
# P1-7 — governance exclusion-strength sweep: exclude fraction = lambda * rho_hat.
#
# For each (setting, lambda) it runs the posterior estimator (05) then the
# weighted training (06), then collects the four requested quantities from the
# resulting posterior/weights:
#
#     test accuracy, noise removal rate, clean retention rate,
#     hard-clean false removal rate
#
# Reruns are skipped when a variant's config hash already has a
# weighted_summary_<tag>.json, so lambda = 0 (soft-only) and lambda = 0.5
# (the frozen "hard50" rule) are often reused rather than recomputed.
#
# Usage:  bash revision/run_p1_lambda_sweep.sh <setting> [<setting> ...]
#   settings: c10_s20  c10_a40  c100_s20  c100_s40  c100_a20
set -uo pipefail
cd /root/quality_noise
PY=.venv/bin/python
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
log() { echo "[$(date +%H:%M:%S)] $*"; }

SETTINGS=("$@")
if [ ${#SETTINGS[@]} -eq 0 ]; then
  SETTINGS=(c10_s20 c10_a40 c100_s20 c100_s40 c100_a20)
fi

# Generate all variant configs up front (idempotent).
$PY revision/gen_lambda_sweep.py --settings "$(IFS=,; echo "${SETTINGS[*]}")" \
    > logs/lambda_gen.log 2>&1 || true
cat logs/lambda_gen.log

for S in "${SETTINGS[@]}"; do
  for LAM in 0.00 0.25 0.50 0.75 1.00; do
    CFG="configs/experiments/lambda_sweep/${S}_lam${LAM}.yaml"
    if [ ! -f "$CFG" ]; then
      log "SKIP $S lambda=$LAM (no config)"
      continue
    fi
    TAG="${S}_lam${LAM}"
    log "=== $S lambda=$LAM ==="

    # --- posterior / weights (05) : cheap, CPU only ---
    if ! ls results/estimator/*/*/*/"weighted_summary_${TAG}.json" >/dev/null 2>&1; then
      log "  fitting estimator (05)"
      $PY scripts/05_fit_estimator.py --exp "$CFG" --tag "$TAG" \
          > "logs/est_${TAG}.log" 2>&1 || { log "  FAILED 05 $TAG"; continue; }
    else
      log "  estimator summary already present"
    fi

    # --- weighted training (06) ---
    log "  weighted training (06)"
    $PY scripts/06_train_weighted.py --exp "$CFG" --tag "$TAG" \
        > "logs/wt_${TAG}.log" 2>&1 || log "  FAILED 06 $TAG"
  done
done

log "=== lambda sweep driver complete ==="
