#!/bin/bash
# 从已导出的 mp4 生成 README 用的 GIF（左：作者权重，右：自训权重；以及自训单独版）。
# 用法（在上游代码目录运行，登录节点上约 1 分钟，只用 CPU）：
#   cd ~/projects/supermimic_mjlab && bash <本仓库路径>/scripts/nscc/make_gifs.sh <输出目录>
# 需要 outputs/videos_author/ 和 outputs/videos_mine/ 都已存在。
set -euo pipefail
OUT=${1:?请给出输出目录}
mkdir -p "$OUT"
FF=$(.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
PAL="split[x][y];[x]palettegen=max_colors=96[p];[y][p]paletteuse=dither=bayer:bayer_scale=4"

for clip in dance2_subject1_wm fallAndGetUp1_subject1 sprint1_subject2 jumps1_subject1; do
  A=outputs/videos_author/$clip.mp4
  M=outputs/videos_mine/$clip.mp4
  # 左右对比：各 400 px 宽，从第 2 秒起取 8 秒，12 fps
  nice "$FF" -y -loglevel error -ss 2 -t 8 -i "$A" -ss 2 -t 8 -i "$M" \
    -filter_complex "[0:v]scale=400:-2,fps=12[a];[1:v]scale=400:-2,fps=12[b];[a][b]hstack=inputs=2,$PAL" \
    "$OUT/${clip}_author_vs_mine.gif"
  # 自训单独版：480 px 宽
  nice "$FF" -y -loglevel error -ss 2 -t 8 -i "$M" \
    -filter_complex "[0:v]scale=480:-2,fps=12,$PAL" \
    "$OUT/${clip}_mine.gif"
done
ls -la "$OUT"
