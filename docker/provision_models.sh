#!/usr/bin/env bash
# 宿主机模型/仓库 provision（在 jy100-08 上跑一次；下载到会被 compose 挂载的目录）
#
# 把 CosyVoice / LatentSync 的仓库与权重放到宿主机，compose 以 volume 挂进容器，
# 避免把 ~10GB 权重与易变的研究仓库塞进镜像层（那样构建慢、难排错、难更新）。
#
# 用法（在仓库根）：bash docker/provision_models.sh
set -euo pipefail

ROOT="${MODELS_ROOT:-$(pwd)/deploy_models}"
mkdir -p "$ROOT"
echo "[provision] 目标目录: $ROOT"

# ── CosyVoice2 ──
if [ ! -d "$ROOT/CosyVoice" ]; then
  echo "[provision] 克隆 CosyVoice ..."
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$ROOT/CosyVoice"
  ( cd "$ROOT/CosyVoice" && git submodule update --init --recursive )
fi
if [ ! -d "$ROOT/CosyVoice/pretrained_models/CosyVoice2-0.5B" ]; then
  echo "[provision] 下载 CosyVoice2-0.5B 权重（~2GB）..."
  pip install -q modelscope 2>/dev/null || true
  python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='$ROOT/CosyVoice/pretrained_models/CosyVoice2-0.5B')" \
    || echo "[warn] CosyVoice2 权重下载失败，请手动放到 $ROOT/CosyVoice/pretrained_models/CosyVoice2-0.5B"
fi

# ── LatentSync ──
if [ ! -d "$ROOT/LatentSync" ]; then
  echo "[provision] 克隆 LatentSync ..."
  git clone https://github.com/bytedance/LatentSync.git "$ROOT/LatentSync"
fi
if [ ! -f "$ROOT/LatentSync/checkpoints/latentsync_unet.pt" ]; then
  echo "[provision] 下载 LatentSync 权重（~5GB，含 whisper）..."
  pip install -q "huggingface_hub[cli]" 2>/dev/null || true
  huggingface-cli download ByteDance/LatentSync-1.5 --local-dir "$ROOT/LatentSync/checkpoints" \
    || echo "[warn] LatentSync 权重下载失败，请按官方 README 放到 $ROOT/LatentSync/checkpoints"
fi

echo "[provision] 完成。目录结构："
echo "  $ROOT/CosyVoice          (仓库 + pretrained_models/CosyVoice2-0.5B)"
echo "  $ROOT/LatentSync         (仓库 + checkpoints/latentsync_unet.pt + whisper)"
echo
echo "下一步：docker compose -f docker/docker-compose.yml build && docker compose -f docker/docker-compose.yml up -d"
