#!/bin/bash
set -e
cmd="
python train_gs.py -s ../../GSexp/data/mip360/kitchen \
    -m output/sparse_gs/kitchen \
    -r 4 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background
"
# --opacity_reset_interval 3000
# TODO: test sh_degree 4
echo $cmd
eval $cmd


# r for resolution
# sparse_view_num for number of views
# visual_hull_4 is the initial point cloud for sparse_id=4
# white_background and random_background are for data augmentation