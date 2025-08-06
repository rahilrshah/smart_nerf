# src/1_create_initial_subset.py

import json
import os
import argparse
from pathlib import Path

def create_initial_subset(dataset_name, full_json_path, num_images, run_name):
    """
    Creates a simple, sparse subset of a dataset.

    Its only two jobs are:
    1. Select every Nth frame to create a smaller, sparser dataset.
    2. Convert the relative image file paths into absolute paths so that
       downstream scripts can find them regardless of the working directory.
    """
    print(f"--- Creating simple sparse subset for '{dataset_name}' ---")

    output_dir = os.path.join("experiments", dataset_name, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created experiment directory: {output_dir}")

    # The directory where the ORIGINAL transforms.json lives. This is the base
    # from which relative image paths are calculated.
    source_base_dir = Path(full_json_path).parent

    with open(full_json_path, 'r') as f:
        full_data = json.load(f)

    frames = full_data.get('frames', [])
    if not frames:
        raise ValueError(f"The source JSON file at '{full_json_path}' contains no frames.")

    # --- Step 1: Select a sparse subset of frames ---
    total_frames = len(frames)
    if num_images >= total_frames:
        step = 1
    else:
        step = total_frames // num_images
    
    subset_frames = frames[::step]

    # --- Step 2: Correct file paths to be absolute ---
    corrected_subset_frames = []
    for frame in subset_frames:
        # Create a copy to avoid modifying the original data structure in memory
        new_frame = frame.copy()
        
        relative_path = Path(new_frame['file_path'])
        
        # Resolve the path relative to the *original* JSON file's location
        # to get the true, absolute path to the image file.
        absolute_path = (source_base_dir / relative_path).resolve()
        
        # Update the frame's file_path entry with the absolute path string
        new_frame['file_path'] = str(absolute_path)
        corrected_subset_frames.append(new_frame)

    print(f"Selected and corrected paths for {len(corrected_subset_frames)} frames from the original {total_frames}.")

    # --- Step 3: Write the new, smaller, corrected dataset file ---
    subset_data = {
        # Copy over essential metadata
        'camera_angle_x': full_data.get('camera_angle_x'),
        'frames': corrected_subset_frames
    }

    output_json_path = os.path.join(output_dir, "transforms.json")
    with open(output_json_path, 'w') as f:
        json.dump(subset_data, f, indent=4)

    print(f"Successfully created initial sparse subset at: {output_json_path}")
    print("--- Initial subset creation complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a sparse subset of a NeRF dataset.")
    parser.add_argument("--dataset", required=True, help="Name of the dataset (e.g., 'fox' or 'lego'). Must match a folder in 'data/'.")
    parser.add_argument("--source", default="data/{dataset}/transforms_train.json", help="Path template to the full source transforms JSON. '{dataset}' will be replaced.")
    parser.add_argument("--num_images", type=int, default=15, help="Approximate number of images to select for the initial sparse set.")
    parser.add_argument("--run_name", default="run_01_geom", help="Name for the first experiment folder.")
    
    args = parser.parse_args()
    
    source_path = args.source.format(dataset=args.dataset)
    
    if not os.path.exists(source_path):
        print(f"❌ ERROR: Source file not found at '{source_path}'. Please ensure the dataset is in the 'data/' directory.")
    else:
        create_initial_subset(args.dataset, source_path, args.num_images, args.run_name)