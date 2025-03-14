#!/bin/bash
set -e

GPU_ids=0
scenes=(leaves fortress fern flower room horns trex orchids)

# <your LLFF path>
data_path=../preprocess_datas/nerf_llff_data

# downsample by 8
res=8
output_path=./output_GSexp_gs_test_LLFF_scenes/LLFF$input_views\_$res\x/


for scene in "${scenes[@]}"
do
    echo ========================= LLFF Train: $scene ========================= 
    CUDA_VISIBLE_DEVICES=$GPU_ids python train.py -s $data_path/$scene -r $res -m $output_path/$scene 

done







# UNIT-TEST
# scene=leaves
# echo ========================= LLFF Train: $scene =========================
# python train.py -s $data_path/$scene -r $res -m $output_path/$scene 




:'
echo ========================= LLFF Train: $scene ========================= 
CUDA_VISIBLE_DEVICES=$GPU_ids python train.py -s $data_path/$scene -r $res -m $output_path/$scene --dataset LLFF \
    --stage train --input_views $input_views --iterations 10000 --densify_until_iter 5000 --total_virtual_num $unseen_num \
    --mvs_config $mvs_config

########
    echo ========================= LLFF Render: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python render.py -s $data_path/$scene -m $output_path/$scene -r $res
    
    echo ========================= LLFF Metric: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python metrics.py -m $output_path/$scene

    echo ========================= LLFF Finish: $scene =========================
########

# downsample by 4
res=4
output_path=./output/LLFF$input_views\_$res\x/
for scene in "${scenes[@]}"
do
    echo ========================= LLFF Train: $scene ========================= 
    CUDA_VISIBLE_DEVICES=$GPU_ids python train.py -s $data_path/$scene -r $res -m $output_path/$scene --dataset LLFF \
    --stage train --input_views $input_views --iterations 10000 --densify_until_iter 5000 --total_virtual_num $unseen_num \
    --mvs_config $mvs_config

    echo ========================= LLFF Render: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python render.py -s $data_path/$scene -m $output_path/$scene -r $res
    
    echo ========================= LLFF Metric: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python metrics.py -m $output_path/$scene

    echo ========================= LLFF Finish: $scene =========================
done


######
######
'