#!/usr/bin/env bash
# 本地 Ubuntu 笔记本：装好上游代码，从 NSCC 拉动作数据，用 CPU 打开 MuJoCo 交互回放窗口。
# 用法：
#   bash setup_and_play.sh                       # 用作者发布的权重
#   bash setup_and_play.sh <policy.pt> <world_model.pt>   # 用自己训练的权重
# 可选环境变量：CLIPS="walk1_subject1 dance1_subject1"  SPEED=1.0  NSCC_HOST=nscc
set -euo pipefail
unset PYTHONPATH                      # ROS 会往 PYTHONPATH 里塞 python3.10 的包，和 uv 环境冲突

REPO=~/supermimic_mjlab
NSCC_HOST=${NSCC_HOST:-nscc}
CLIPS=${CLIPS:-"walk1_subject1 dance1_subject1"}
SPEED=${SPEED:-1.0}

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

if [ ! -d "$REPO" ]; then
  git clone https://github.com/Breezewrf/supermimic_mjlab "$REPO"
fi
cd "$REPO"
echo "上游 commit: $(git log -1 --format='%h %ci')"
uv sync

NPZ=src/assets/motions/g1/lafan1_npz
if [ "$(ls $NPZ/*.npz 2>/dev/null | wc -l)" -ne 40 ]; then
  mkdir -p "$NPZ"
  rsync -avP "$NSCC_HOST":~/projects/supermimic_mjlab/$NPZ/ "$NPZ"/
fi
echo "本地 NPZ 数量: $(ls $NPZ/*.npz | wc -l)"

POLICY=${1:-checkpoints/best/policy_best.pt}
WM=${2:-checkpoints/best/world_model_best.pt}
echo "策略: $POLICY"; echo "世界模型: $WM"

# shellcheck disable=SC2086
.venv/bin/python scripts/play_supermimic.py \
  --device cpu --checkpoint "$POLICY" --world-model "$WM" \
  --clips $CLIPS --loop --speed "$SPEED" \
  --motion-dir "$NPZ"
