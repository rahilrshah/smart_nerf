# src/create_subset.py

import json
import os
import argparse
import shutil
from pathlib import Path # Use the modern, robust pathlib

def create_initial_subset(dataset_name, full_json_path, num_images, run_name):
    """
    Creates the first sparse dataset with corrected, absolute image file paths.
    """
    print(f"--- Creating initial subset for '{dataset_name}' ---")

    output_dir = os.path.join("experiments", dataset_name, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created experiment directory: {output_dir}")

    # The directory where the ORIGINAL transforms.json lives, which is our base for relative paths
    source_base_dir = Path(full_json_path).parent

    with open(full_json_path, 'r') as f:
        full_data = json.load(f)

    frames = full_data['frames']
    if not frames:
        raise ValueError("The source JSON file contains no frames.")

    total_frames = len(frames)
    step = max(1, total_frames // num_images)
    subset_frames_info = frames[::step]
    
    # --- MODIFIED: Rewrite file paths to be absolute ---
    corrected_subset_frames = []
    for frame in subset_frames_info:
        # Get the relative path from the JSON (e.g., './train/r_0.png')
        relative_path = Path(frame['file_path'])
        
        # Join it with the original JSON's directory to get the true path
        # And resolve it to a clean, absolute path string
        absolute_path_str = str((source_base_dir / relative_path).resolve())
        
        # Update the frame's file_path entry
        frame['file_path'] = absolute_path_str
        corrected_subset_frames.append(frame)

    print(f"Selected and corrected paths for {len(corrected_subset_frames)} frames out of {total_frames}")

    subset_data = {
        'camera_angle_x': full_data.get('camera_angle_x'),
        'frames': corrected_subset_frames
    }

    output_json_path = os.path.join(output_dir, "transforms.json")
    with open(output_json_path, 'w') as f:
        json.dump(subset_data, f, indent=4)

    print(f"Successfully created initial subset at: {output_json_path}")
    print("--- Initial subset creation complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an initial sparse subset of a NeRF dataset.")
    parser.add_argument("--dataset", required=True, help="Name of the dataset (e.g., 'lego').")
    parser.add_argument("--source", default="data/{dataset}/transforms_train.json", help="Path template to the full transforms JSON.")
    parser.add_argument("--num_images", type=int, default=15, help="Approximate number of images for the initial sparse set.")
    parser.add_argument("--run_name", default="run_01_initial", help="Name for the first experiment folder.")
    args = parser.parse_args()
    source_path = args.source.format(dataset=args.dataset)
    if not os.path.exists(source_path):
        print(f"Error: Source file not found at {source_path}")
    else:
        create_initial_subset(args.dataset, source_path, args.num_images, args.run_name)