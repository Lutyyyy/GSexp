#!/bin/bash
set -e
# -r is for resolution
# cmd="
# python train_gs.py -s data/mip360/kitchen \
#     -m output/dense_gs/kitchen \
#     -r 4 --sh_degree 2 \
#     --init_pcd_name visual_hull_4 \
#     --white_background --random_background \
#     --iterations 30000 --densify_until_iter 15000 \
#     -- opacity_reset_interval 3000 \
#     --eval
# "  # cmd for dense GS
cmd="
python train_gs.py -s data/mip360/kitchen \
    -m output/coarse_gs_test/kitchen \
    -r 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background \
    --eval
"
echo $cmd
eval $cmd
