#!/usr/bin/env bash
# GPU 1 queue: qwen3.6-27b -> gemma-4-31b-it
# Usage: CUDA_VISIBLE_DEVICES=1 bash run_1.sh
#   CONDITIONS=statute_swap LOGTAG=_swap CUDA_VISIBLE_DEVICES=1 bash run_1.sh   # 반사실 조건만
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p logs

PORT="${PORT:-8002}"
# "hf_repo|safe_name|serve extra args|runner extra args"
MODELS=(
  "Qwen/Qwen3.6-27B|qwen3.6-27b||--no-think"
  "google/gemma-4-31B-it|gemma-4-31b-it||"
)

SPID=""
trap 'kill 0' INT TERM
trap '[ -n "$SPID" ] && kill "$SPID" 2>/dev/null || true' EXIT

for spec in "${MODELS[@]}"; do
  IFS='|' read -r repo safe extra rextra <<< "$spec"
  # ONLY=name1,name2 로 큐의 일부 모델만 실행
  if [ -n "${ONLY:-}" ] && [[ ",${ONLY}," != *",${safe},"* ]]; then continue; fi

  echo "=== $(date '+%H:%M:%S') loading $repo (a few minutes) ==="
  # shellcheck disable=SC2086
  vllm serve "$repo" --port "$PORT" --max-model-len 24576 --max-num-seqs 64 \
    --gpu-memory-utilization "${GPU_UTIL:-0.92}" $extra > "logs/serve_${safe}.log" 2>&1 &
  SPID=$!
  until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    kill -0 "$SPID" 2>/dev/null || { echo "vLLM died — see logs/serve_${safe}.log"; exit 1; }
    sleep 5
  done

  echo "=== $(date '+%H:%M:%S') running $safe ==="
  # shellcheck disable=SC2086
  python scripts/run_vllm.py --model "$repo" --model-safe "$safe" --port "$PORT" \
    --max-tokens "${MAX_TOKENS:-2048}" --conditions "${CONDITIONS:-all}" \
    --concurrency "${CONCURRENCY:-16}" $rextra 2>&1 \
    | tee "logs/run_${safe}${LOGTAG:-}.log"

  kill "$SPID" 2>/dev/null || true
  wait "$SPID" 2>/dev/null || true

  # 모델 완료 시점에 해당 모델 결과만 커밋 (동시 실행 대비 flock, push 실패 시 로컬 커밋 유지)
  (
    flock 9
    git add "results/${safe}__"*.jsonl 2>/dev/null
    git commit -m "results: ${safe} (${CONDITIONS:-all})" >/dev/null 2>&1 \
      && { git push origin HEAD:experiment/judgment-prediction >/dev/null 2>&1 \
           || echo "[git] push rejected — commit kept locally (${safe})"; } \
      || true
  ) 9>/tmp/kldbench-git.lock

  sleep 10
done
echo "=== GPU1 queue done $(date '+%H:%M:%S') ==="
