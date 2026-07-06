# SDCL + CGLF（SYSU 训练/测试 batch 均为 -b，例如 96 为显存上限）
# --cglf-threshold: 0.0=无过滤; 0.25 为此前较优 CGLF
# --grad-accum-steps: 例如 2 则每 2 个 iter 一次 step（显存仍为 -b）
# --cross-modal-mode: alternating(默认) | rgb2ir | both
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0,1 python sdcl_sysu_originalmain_rcl_cwsl_v2_fix1.py -b 96 -a agw -d sysu_all --iters 200 --epochs 50 --momentum 0.1 --eps 0.6 --num-instances 16 --cglf-threshold 0.00 --best-select-mode full --stage1-best-select-mode legacy --grad-accum-steps 1
