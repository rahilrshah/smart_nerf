# src/4_run_neuralangelo.py

import argparse
import subprocess
import os
import json
import numpy as np
from pathlib import Path
import glob

# NOTE: This script requires 'imageio' which is in your requirements.txt
import imageio.v2 as imageio

def prepare_data_for_neuralangelo(dataset_name, optimized_json_path, output_data_dir):
    """
    Converts instant-ngp's transforms.json into the data format
    expected by Neuralangelo (a folder with images, masks, and poses).
    """
    print(f"--- Preparing data for Neuralangelo in '{output_data_dir}' ---")
    
    img_dir = os.path.join(output_data_dir, "images")
    pose_dir = os.path.join(output_data_dir, "poses")
    # For synthetic data, we can create trivial all-white masks.
    # For real data, providing real masks would be much better.
    mask_dir = os.path.join(output_data_dir, "masks")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    with open(optimized_json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Preparing {len(data['frames'])} images...")
    for frame in data['frames']:
        # The paths in our optimized json are already absolute
        src_img_path = frame['file_path']
        img_filename = os.path.basename(src_img_path)
        dst_img_path = os.path.join(img_dir, img_filename)

        if not os.path.exists(dst_img_path):
             # Use a symlink if possible (faster), otherwise copy.
             try:
                os.link(src_img_path, dst_img_path)
             except Exception: # Catch broader exceptions for cross-drive or permission issues
                import shutil
                shutil.copy(src_img_path, dst_img_path)

        # Write camera pose matrix to a .txt file
        pose_filename = os.path.splitext(img_filename)[0] + ".txt"
        pose = np.array(frame['transform_matrix'])
        np.savetxt(os.path.join(pose_dir, pose_filename), pose)

        # Create a trivial all-white mask
        mask_filename = os.path.splitext(img_filename)[0] + ".png" # Masks should be png
        # We need the image dimensions to create the mask
        try:
            img_for_size = imageio.imread(src_img_path)
            h, w, _ = img_for_size.shape
            mask = np.ones((h, w), dtype=np.uint8) * 255
            imageio.imwrite(os.path.join(mask_dir, mask_filename), mask)
        except Exception as e:
            print(f"Warning: Could not create mask for {src_img_path}. Error: {e}")

    print("--- Data preparation complete ---")


def run_neuralangelo_pipeline(optimized_json_path, output_base_dir):
    """
    Orchestrates the full Neuralangelo pipeline: data prep, training, meshing.
    """
    dataset_name = Path(optimized_json_path).parent.name
    
    # Define working directories for Neuralangelo
    neuralangelo_data_dir = os.path.abspath(os.path.join(output_base_dir, dataset_name, "neuralangelo_data"))
    neuralangelo_output_dir = os.path.abspath(os.path.join(output_base_dir, dataset_name, "neuralangelo_output"))

    # Step 1: Convert our optimized data to the format Neuralangelo needs.
    prepare_data_for_neuralangelo(dataset_name, optimized_json_path, neuralangelo_data_dir)

    # Step 2: Launch Neuralangelo training.
    print("\n--- Starting Neuralangelo Training (This will take a long time) ---")
    
    # ASSUMPTION: You have a custom config file for your dataset
    # e.g., vendor/neuralangelo/projects/neuralangelo/configs/custom/hotdog.yaml
    # This config should point to the correct base config for blender-style data.
    config_file = f"projects/neuralangelo/configs/custom/{dataset_name}.yaml"
    
    training_cmd = [
        "python", "projects/neuralangelo/train.py",
        f"--config={config_file}",
        f"data.dir={neuralangelo_data_dir}", # Use dot notation to override config values
        f"log_dir={neuralangelo_output_dir}",
    ]
    # This command must be run from inside the vendor directory
    subprocess.run(training_cmd, cwd="vendor/neuralangelo", check=True)
    print("--- Training complete ---")

    # Step 3: Launch Mesh Extraction
    print("\n--- Starting Neuralangelo Mesh Extraction ---")
    try:
        # Find the config file of the completed training run to pass to the mesher.
        trained_config = sorted(glob.glob(f"{neuralangelo_output_dir}/**/config.yaml", recursive=True))[-1]
    except IndexError:
        print("❌ ERROR: Could not find trained config.yaml for meshing. Halting.")
        exit()
        
    meshing_cmd = [
        "python", "projects/neuralangelo/extract_mesh.py",
        f"--config={trained_config}",
        f"log_dir={neuralangelo_output_dir}" # Specify log_dir again for consistency
    ]
    subprocess.run(meshing_cmd, cwd="vendor/neuralangelo", check=True)
    print(f"--- Meshing complete. Find your final .obj file in '{neuralangelo_output_dir}'. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Neuralangelo pipeline on an optimized dataset.")
    parser.add_argument("--optimized_json", required=True)
    
    # --- THE FINAL, DEFINITIVE FIX IS HERE ---
    # Add the missing --output_base argument to the parser.
    parser.add_argument("--output_base", default="results", help="Base directory to save final outputs.")
    
    args = parser.parse_args()

    if not os.path.exists(args.optimized_json):
        print(f"Error: Optimized JSON file not found at {args.optimized_json}")
    else:
        run_neuralangelo_pipeline(args.optimized_json, args.output_base)