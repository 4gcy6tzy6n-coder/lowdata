#!/usr/bin/env bash
# Restart just the CIFAR-100 asymmetric-40% leg of Wave C.
#
# The original Wave C died here with `OSError: [Errno 28] No space left on
# device` when the container filled up (pip cache had grown to 5.5 GB).  The
# half-written run directories were removed, so this re-runs the leg cleanly.
set -uo pipefail
cd /root/quality_noise
PY=.venv/bin/python
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }

EXP=configs/experiments/_rev_c100_asym40_5seeds.yaml
TAG=c100a40

df -h / | tail -1

log "=== traces: $TAG (5 runs) ==="
pids=()
for i in 0 1 2 3 4; do
  $PY scripts/02_collect_traces.py --exp "$EXP" --runs "$i" > "logs/trace_${TAG}_r${i}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
log "traces done; failures:"
grep -l -E "Traceback|Error" logs/trace_${TAG}_r*.log 2>/dev/null || echo "  none"

log "=== quality: $TAG ==="
for i in 0 1 2 3 4; do
  $PY scripts/03_compute_quality.py --exp "$EXP" --runs "$i" > "logs/qual_${TAG}_r${i}.log" 2>&1 &
done
wait

log "=== detectability: $TAG ==="
for i in 0 1 2 3 4; do
  $PY scripts/04_detectability.py --exp "$EXP" --runs "$i" > "logs/det_${TAG}_r${i}.log" 2>&1 &
done
wait

log "=== $TAG complete ==="
for i in 0 1 2 3 4; do printf "  seed%s: " "$i"; ls results/traces/cifar100/asymmetric0.4/seed$i 2>/dev/null | wc -l; done
df -h / | tail -1
