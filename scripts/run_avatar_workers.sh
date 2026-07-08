#!/usr/bin/env bash
# 启动 N 个常驻 LatentSync worker（每卡一个进程，模型只加载一次）
#
# 用法：
#   LATENTSYNC_DIR=/path/to/LatentSync bash scripts/run_avatar_workers.sh [N] [BASE_PORT]
#   N          worker 数（默认 8，对应 8×3090）
#   BASE_PORT  起始端口（默认 8010，依次 8010..8010+N-1）
#
# 停止：bash scripts/run_avatar_workers.sh stop
set -euo pipefail

N="${1:-8}"
BASE_PORT="${2:-8010}"
PIDDIR="${PIDDIR:-./workspace_data/worker_pids}"
mkdir -p "$PIDDIR"

if [ "${1:-}" = "stop" ]; then
    for f in "$PIDDIR"/worker_*.pid; do
        [ -f "$f" ] || continue
        kill "$(cat "$f")" 2>/dev/null || true
        rm -f "$f"
    done
    echo "已停止所有 worker"
    exit 0
fi

: "${LATENTSYNC_DIR:?请设置 LATENTSYNC_DIR 指向 LatentSync 仓库根}"

for i in $(seq 0 $((N-1))); do
    port=$((BASE_PORT + i))
    CUDA_VISIBLE_DEVICES="$i" \
        nohup python -m krvoiceai.api.avatar_worker --port "$port" \
        > "$PIDDIR/worker_${i}.log" 2>&1 &
    echo $! > "$PIDDIR/worker_${i}.pid"
    echo "worker $i  GPU=$i  port=$port  pid=$!  log=$PIDDIR/worker_${i}.log"
done

echo
echo "已启动 $N 个 worker。健康检查："
for i in $(seq 0 $((N-1))); do
    echo "  curl -s localhost:$((BASE_PORT + i))/health"
done
