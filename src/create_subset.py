# src/1_create_initial_subset.py

import json
import os
import argparse
import shutil

def create_initial_subset(dataset_name, full_json_path, num_images, run_name):
    """
    Creates the first sparse dataset for analysis from the full dataset.
    """
    print(f"--- Creating initial subset for '{dataset_name}' ---")

    # Define the output directory for this first experimental run
    output_dir = os.path.join("experiments", dataset_name, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created experiment directory: {output_dir}")

    # Load the full dataset JSON
    with open(full_json_path, 'r') as f:
        full_data = json.load(f)

    frames = full_data['frames']
    if not frames:
        raise ValueError("The source JSON file contains no frames.")

    # Calculate the step to get the desired number of images
    total_frames = len(frames)
    step = max(1, total_frames // num_images)
    
    # Select frames using slicing
    subset_frames = frames[::step]

    print(f"Selected {len(subset_frames)} frames out of {total_frames} (using a step of {step})")

    # Create the new data structure for the subset
    subset_data = {
        'camera_angle_x': full_data.get('camera_angle_x'),
        'frames': subset_frames
    }

    # Write the new transforms.json for the subset
    output_json_path = os.path.join(output_dir, "transforms.json")
    with open(output_json_path, 'w') as f:
        json.dump(subset_data, f, indent=4)

    print(f"Successfully created initial subset at: {output_json_path}")
    print("--- Initial subset creation complete ---")
    return output_json_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an initial sparse subset of a NeRF dataset.")
    parser.add_argument("--dataset", required=True, help="Name of the dataset (e.g., 'lego').")
    parser.add_argument("--source", default="data/{dataset}/transforms_train.json",
                        help="Path template to the full transforms JSON. '{dataset}' will be replaced.")
    parser.add_argument("--num_images", type=int, default=15, help="Approximate number of images for the initial sparse set.")
    parser.add_argument("--run_name", default="run_01_initial", help="Name for the first experiment folder.")
    
    args = parser.parse_args()

    # Format the source path with the dataset name
    source_path = args.source.format(dataset=args.dataset)

    if not os.path.exists(source_path):
        print(f"Error: Source file not found at {source_path}")
        print("Please ensure your data is structured correctly in the 'data/' directory.")
    else:
        create_initial_subset(args.dataset, source_path, args.num_images, args.run_name)