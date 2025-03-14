from PIL import Image
import os
from pathlib import Path
import glob

def downsample_images(input_dir, scale_factors: list):
    # Create output directory name (e.g., "images_4" for 4x downsampling)
    for scale_factor in scale_factors:
        parent_dir = str(Path(input_dir).parent)
        output_dir = os.path.join(parent_dir, f"images_{scale_factor}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = glob.glob(os.path.join(input_dir, "*.[pjJ][npP][gG]"))  # matches .jpg, .png, .jpeg
        print(f"Found {len(image_files)} images in {input_dir}")
        
        for idx, img_path in enumerate(sorted(image_files)):
            # Open image
            img = Image.open(img_path)
            
            # Calculate new dimensions
            new_width = img.width // scale_factor
            new_height = img.height // scale_factor
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save with new name
            new_name = f"image{idx:03d}.png"
            output_path = os.path.join(output_dir, new_name)
            resized_img.save(output_path)
            
            print(f"Processed {img_path} -> {output_path}")

if __name__ == "__main__":
    # Example usage
    scenes = ["horns", "trex", "fortress", "flower", 'leaves', 'room', 'orchids', 'fern']
    scale_factors = [4, 8]    # Change this to your desired downsampling factor

    for input_dir in [f"/root/autodl-tmp/preprocess_datas/nerf_llff_data/{scene}/images" for scene in scenes]:
        downsample_images(input_dir, scale_factors)