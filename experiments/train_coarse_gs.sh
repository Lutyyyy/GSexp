#!/bin/bash
set -e
# -r is for resolution
cmd="
python train_gs.py -s data/mip360/kitchen \
    -m output/coarse_gs/kitchen \
    -r 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background \
    --eval
"
echo $cmd
eval $cmd
