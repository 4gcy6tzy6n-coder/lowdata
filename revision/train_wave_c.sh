#!/usr/bin/env bash
# Wave C — the three CIFAR-100 settings that exist only as archived summaries.
#
# Gated on the reproduction check: run only after
#   python revision/check_reproduction_gate.py   ->   GATE: PASS
# which confirms the pipeline reproduces the frozen C100-S20 headline and that
# the old (seeds 0-4) and new (seeds 5-9) halves agree.
#
# Everything produced here is tagged as a revision rerun; archived summaries are
# kept for provenance cross-check only and are never merged into these numbers.
set -uo pipefail
cd /root/quality_noise
PY=.venv/bin/python
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
log() { echo "[$(date +%H:%M:%S)] $*"; }

wave () {
  local exp="$1"; local tag="$2"
  log "=== traces: $tag ($exp) ==="
  local pids=()
  for i in 0 1 2 3 4; do
    $PY scripts/02_collect_traces.py --exp "$exp" --runs "$i" > "logs/trace_${tag}_r${i}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  log "=== quality: $tag ==="
  for i in 0 1 2 3 4; do
    $PY scripts/03_compute_quality.py --exp "$exp" --runs "$i" > "logs/qual_${tag}_r${i}.log" 2>&1 &
  done
  wait
  log "=== detectability: $tag ==="
  for i in 0 1 2 3 4; do
    $PY scripts/04_detectability.py --exp "$exp" --runs "$i" > "logs/det_${tag}_r${i}.log" 2>&1 &
  done
  wait
  log "=== $tag complete ==="
}

log "=== Wave C start (C100 S40/A20/A40 revision reruns) ==="
wave configs/experiments/_rev_c100_sym40_5seeds.yaml  c100s40
wave configs/experiments/_rev_c100_asym20_5seeds.yaml c100a20
wave configs/experiments/_rev_c100_asym40_5seeds.yaml c100a40
log "=== WAVE C COMPLETE ==="
