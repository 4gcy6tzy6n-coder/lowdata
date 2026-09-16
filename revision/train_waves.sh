#!/usr/bin/env bash
# Revision training driver (single GPU, CPU-parallel prepare running alongside).
#
# Wave A: CIFAR-100 symmetric-20% seeds 5-9
#         -> takes the P0-2 reversal setting from 5 to 10 seeds.
# Wave B: CIFAR-100N human-annotation noise seeds 0-4 (P0/P1-5 real-noise extension).
#
# Within a wave the five runs are launched simultaneously (a ResNet-18 at batch
# 128 uses only a few GB, so five concurrent runs fill the 5090 without
# oversubscribing the data-loader CPU budget), then quality/detectability are
# computed for the new runs.
set -uo pipefail
cd /root/quality_noise
PY=.venv/bin/python
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs
log() { echo "[$(date +%H:%M:%S)] $*"; }

wave_traces () {
  local exp="$1"; local tag="$2"; local n="$3"
  local pids=()
  for i in $(seq 0 $((n-1))); do
    $PY scripts/02_collect_traces.py --exp "$exp" --runs "$i" \
        > "logs/trace_${tag}_r${i}.log" 2>&1 &
    pids+=($!)
  done
  log "launched ${#pids[@]} training runs for $tag (pids: ${pids[*]})"
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  log "wave $tag training finished (rc=$rc)"
  return $rc
}

log "=== revision training start ==="

# ---------------- Wave A: C100-S20 seeds 5..9 ----------------
wave_traces configs/experiments/gate_a_cifar100_sym20_seeds5to9.yaml c100s5to9 5
log "=== quality: cifar100 sym20 (seeds 5-9) ==="
for i in 0 1 2 3 4; do
  $PY scripts/03_compute_quality.py --exp configs/experiments/gate_a_cifar100_sym20_seeds5to9.yaml --runs "$i" \
    > "logs/qual_c100_r${i}.log" 2>&1 &
done
wait
log "wave A complete"

# ---------------- Wave B: CIFAR-100N seeds 0..4 ----------------
wave_traces configs/experiments/gate_a_cifar100n_5seeds.yaml c100n 5
log "=== quality: cifar100n ==="
for i in 0 1 2 3 4; do
  $PY scripts/03_compute_quality.py --exp configs/experiments/gate_a_cifar100n_5seeds.yaml --runs "$i" \
    > "logs/qual_c100n_r${i}.log" 2>&1 &
done
wait
log "=== detectability: cifar100n ==="
for i in 0 1 2 3 4; do
  $PY scripts/04_detectability.py --exp configs/experiments/gate_a_cifar100n_5seeds.yaml --runs "$i" \
    > "logs/det_c100n_r${i}.log" 2>&1 &
done
wait
log "=== detectability: cifar100 sym20 (10 seeds) ==="
for i in 0 1 2 3 4 5 6 7 8 9; do
  $PY scripts/04_detectability.py --exp configs/experiments/gate_a_cifar100_sym20_10seeds.yaml --runs "$i" \
    > "logs/det_c100_r${i}.log" 2>&1 &
done
wait
log "=== ALL REVISION TRAINING WAVES COMPLETE ==="
