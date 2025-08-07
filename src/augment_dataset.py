# src/3_augment_dataset.py (DIAGNOSTIC VERSION)

import json
import os
import argparse
from pathlib import Path

def augment_dataset(previous_run_path, new_run_name, full_dataset_json_path):
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
        recommended_absolute_paths = {line.strip() for line in f if line.strip()} # Ignore empty lines

    print(f"Loaded {len(recommended_absolute_paths)} new images to add.")
    
    # --- DEBUG PRINTS FOR AUGMENTATION ---
    print("\n--- [DEBUG] augment_dataset.py ---")
    print(f"Number of frames in previous dataset: {len(previous_data['frames'])}")
    print(f"Number of recommended paths read from file: {len(recommended_absolute_paths)}")
    print("Sample of recommended paths read:")
    for p in list(recommended_absolute_paths)[:3]:
        print(f"  - {p}")
    
    # Start with a map of the frames we already have, keyed by absolute path.
    current_frames_map = {frame['file_path']: frame for frame in previous_data['frames']}

    # Create a lookup map of the full dataset from ABSOLUTE path -> frame data.
    source_base_dir = Path(full_dataset_json_path).parent
    absolute_path_map = {}
    for frame in full_data['frames']:
        abs_path = str((source_base_dir / Path(frame['file_path'])).resolve())
        absolute_path_map[abs_path] = frame

    print(f"Built original dataset lookup map with {len(absolute_path_map)} entries.")

    # Add the new recommended frames to our map.
    added_count = 0
    for path_to_add in recommended_absolute_paths:
        if path_to_add not in current_frames_map: # Only add if it's new
            if path_to_add in absolute_path_map:
                new_frame_data = absolute_path_map[path_to_add].copy()
                new_frame_data['file_path'] = path_to_add
                current_frames_map[path_to_add] = new_frame_data
                added_count += 1
            else:
                print(f"  - WARNING: Recommended path '{path_to_add}' could not be found in the original dataset map.")
        else:
             print(f"  - INFO: Path '{path_to_add}' already in dataset. Skipping.")

    final_frames = list(current_frames_map.values())
    
    print(f"Successfully added {added_count} new unique frames.")
    print("--- [DEBUG] End of Augmentation ---")

    subset_data = {
        'camera_angle_x': full_data.get('camera_angle_x'),
        'frames': final_frames
    }

    print(f"New augmented dataset has {len(final_frames)} frames.")

    output_json_path = os.path.join(new_run_dir, "transforms.json")
    with open(output_json_path, 'w') as f:
        json.dump(subset_data, f, indent=4)
        
    print(f"Successfully created augmented dataset at: {output_json_path}")
    print("--- Augmentation complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment a dataset with recommended images.")
    parser.add_argument("--previous_run", required=True)
    parser.add_argument("--new_run_name", required=True)
    parser.add_argument("--source", default="data/{dataset}/transforms_train.json")
    args = parser.parse_args()
    
    dataset_name = Path(args.previous_run).parts[1]
    full_json_path = args.source.format(dataset=dataset_name)

    augment_dataset(args.previous_run, args.new_run_name, full_json_path)