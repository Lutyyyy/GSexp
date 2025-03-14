#!/bin/bash
# filepath: process_nerf_data.sh

# Base directories
INPUT_BASE=$(eval echo "~/autodl-tmp/datas/nerf_llff_data")
OUTPUT_BASE=$(eval echo "~/autodl-tmp/preprocess_datas/nerf_llff_data")

# Create output base directory if it doesn't exist
mkdir -p "$OUTPUT_BASE"

for scene in $(find "$INPUT_BASE" -maxdepth 1 -mindepth 1 -type d); do
    # Get the scene name (basename of the directory)
    scene_name=$(basename "$scene")
    
    echo "Processing scene: $scene_name"
    
    # Create output directory for this scene
    mkdir -p "$OUTPUT_BASE/$scene_name"
    
    # 1. Run COLMAP image undistorter
    colmap image_undistorter \
        --image_path "$INPUT_BASE/$scene_name/images" \
        --input_path "$INPUT_BASE/$scene_name/sparse/0" \
        --output_path "$OUTPUT_BASE/$scene_name" \
        --output_type COLMAP
    
    # 2. Remove the generated sparse folder
    mkdir -p "$OUTPUT_BASE/$scene_name/sparse/0"
    
    # 3. Copy the original sparse folder
    mv "$OUTPUT_BASE/$scene_name/sparse/"*.bin "$OUTPUT_BASE/$scene_name/sparse/0/"
    
    echo "Completed processing $scene_name"
done

echo "All scenes processed successfully"