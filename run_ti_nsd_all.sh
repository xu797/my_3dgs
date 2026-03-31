#!/usr/bin/env bash
set -euo pipefail

# 批量运行 TI-NSD 所有一级场景：train -> render -> metrics
# 用法：
#   bash run_ti_nsd_all.sh
#   bash run_ti_nsd_all.sh data/TI-NSD output/ti-nsd

DATA_ROOT="${1:-data/TI-NSD}"
OUTPUT_ROOT="${2:-output/ti-nsd}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] 数据目录不存在: $DATA_ROOT"
  exit 1
fi

# 先确保总输出目录存在
mkdir -p "$OUTPUT_ROOT"

echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] OUTPUT_ROOT=$OUTPUT_ROOT"

scene_count=0

for scene_path in "$DATA_ROOT"/*; do
  [[ -d "$scene_path" ]] || continue

  scene_name="$(basename "$scene_path")"
  model_path="$OUTPUT_ROOT/$scene_name"

  # 每个场景开始前确保对应输出目录已创建
  mkdir -p "$model_path"

  scene_count=$((scene_count + 1))

  echo ""
  echo "=============================="
  echo "[SCENE] $scene_name"
  echo "[PATH ] source=$scene_path"
  echo "[PATH ] model=$model_path"
  echo "=============================="

  echo "[STEP] train"
  python train.py -s "$scene_path" -m "$model_path" --eval

  echo "[STEP] render (test only)"
  python render.py -m "$model_path" -s "$scene_path" --skip_train

  echo "[STEP] metrics"
  python metrics.py -m "$model_path"
done

if [[ "$scene_count" -eq 0 ]]; then
  echo "[WARN] 在 $DATA_ROOT 下没有找到一级场景目录"
  exit 1
fi

echo ""
echo "[DONE] 已完成 $scene_count 个场景"
