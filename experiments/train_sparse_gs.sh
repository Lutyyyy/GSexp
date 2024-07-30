#!/bin/bash
set -e
cmd="
python train_gs.py -s data/mip360/kitchen \
    -m output/sparse_gs/kitchen \
    -r 4 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background
"
echo $cmd
eval $cmd
