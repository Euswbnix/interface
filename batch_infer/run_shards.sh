#!/usr/bin/env bash
# 一键启动 N 个分片并行推理。在仓库根目录(interface)下运行：
#   bash batch_infer/run_shards.sh [片数N] [视频目录]
# 例：bash batch_infer/run_shards.sh 10 /home/smartlink_temp/workspace/videos
set -euo pipefail

N="${1:-10}"
VIDEO_DIR="${2:-/home/smartlink_temp/workspace/videos}"
MANIFEST="batch_infer/manifest_5.1-5.20.csv"
OUT="batch_infer/results_5.1-5.20.jsonl"

echo "启动 ${N} 个分片，视频目录 ${VIDEO_DIR}"
mkdir -p batch_infer/shard_logs
for ((i=0; i<N; i++)); do
  nohup python batch_infer/run_infer.py \
    --manifest "${MANIFEST}" \
    --video-dir "${VIDEO_DIR}" \
    --out "${OUT}" \
    --num-shards "${N}" --shard-id "${i}" \
    --print-every 200 \
    > "batch_infer/shard_logs/shard${i}.log" 2>&1 &
  echo "  分片 ${i} 已启动 PID=$!"
  sleep 2   # 错开模型加载，避免同一瞬间峰值显存叠加
done
echo "全部启动。日志在 batch_infer/shard_logs/shard*.log"
echo "看总体进度： tail -f batch_infer/shard_logs/shard0.log"
echo "看显存/利用率： watch -n1 nvidia-smi"
