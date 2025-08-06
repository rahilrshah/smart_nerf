# src/3_augment_dataset.py

import json
import os
import argparse
from pathlib import Path

def augment_dataset(previous_run_path, new_run_name, full_dataset_json_path):
    """
    Creates a new, augmented dataset based on the recommendations from a previous run.
    Assumes all file paths are absolute.
    """
    print(f"--- Augmenting dataset from run: {previous_run_path} ---")

    recommendations_path = os.path.join(previous_run_path, "recommended_images.txt")
    if not os.path.exists(recommendations_path):
        print(f"Error: No recommendations file found at {recommendations_path}."); return

    dataset_name = Path(previous_run_path).parts[1]
    new_run_dir = os.path.join("experiments", dataset_name, new_run_name)
    os.makedirs(new_run_dir, exist_ok=True)
    print(f"Created new experiment directory: {new_run_dir}")

    # Load all necessary data
    with open(os.path.join(previous_run_path, "transforms.json"), 'r') as f:
        previous_data = json.load(f)
    with open(full_dataset_json_path, 'r') as f:
        full_data = json.load(f)
    with open(recommendations_path, 'r') as f:
        # These are now absolute paths, same as in previous_data
        recommended_absolute_paths = {line.strip() for line in f}

    print(f"Loaded {len(recommended_absolute_paths)} new images to add.")

    # --- THE SIMPLIFIED LOGIC ---
    # Create a simple lookup map from absolute path -> frame data.
    source_base_dir = Path(full_dataset_json_path).parent
    absolute_path_map = {}
    for frame in full_data['frames']:
        abs_path = str((source_base_dir / Path(frame['file_path'])).resolve())
        absolute_path_map[abs_path] = frame

    # Start with the frames from the previous run
    new_frames = previous_data['frames']
    
    # Add the new recommended frames
    for path in recommended_absolute_paths:
        if path in absolute_path_map:
            # We must create a copy and ensure its path is also absolute
            new_frame_data = absolute_path_map[path].copy()
            new_frame_data['file_path'] = path 
            new_frames.append(new_frame_data)
        else:
            print(f"Warning: Recommended path '{path}' not found.")
            
    # Remove duplicates by checking the absolute file path (already done by using a set)
    seen_paths = set()
    unique_new_frames = []
    for frame in new_frames:
        if frame['file_path'] not in seen_paths:
            unique_new_frames.append(frame)
            seen_paths.add(frame['file_path'])

    subset_data = {
        'camera_angle_x': full_data.get('camera_angle_x'),
        'frames': unique_new_frames
    }

    print(f"New augmented dataset has {len(subset_data['frames'])} frames.")

    output_json_path = os.path.join(new_run_dir, "transforms.json")
    with open(output_json_path, 'w') as f: json.dump(subset_data, f, indent=4)
        
    print(f"Successfully created augmented dataset at: {output_json_path}")
    print("--- Augmentation complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment a dataset with recommended images.")
    parser.add_argument("--previous_run", required=True)
    parser.add_argument("--new_run_name", required=True)
    parser.add_argument("--source", default="data/{dataset}/transforms_train.json")
    args = parser.parse_args()
    
    # Use pathlib here as well for robustness
    dataset_name = Path(args.previous_run).parts[1]
    full_json_path = args.source.format(dataset=dataset_name)

    augment_dataset(args.previous_run, args.new_run_name, full_json_path)