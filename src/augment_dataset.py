# src/3_augment_dataset.py

import json
import os
import argparse
import glob

def augment_dataset(previous_run_path, new_run_name, full_dataset_json_path):
    """
    Creates a new, augmented dataset based on the recommendations from a previous run.
    """
    print(f"--- Augmenting dataset from run: {previous_run_path} ---")

    recommendations_path = os.path.join(previous_run_path, "recommended_images.txt")
    if not os.path.exists(recommendations_path):
        print(f"Error: No recommendations file found at {recommendations_path}. Cannot augment.")
        return

    # Create the new experiment directory
    dataset_name = previous_run_path.split(os.sep)[1]
    new_run_dir = os.path.join("experiments", dataset_name, new_run_name)
    os.makedirs(new_run_dir, exist_ok=True)
    print(f"Created new experiment directory: {new_run_dir}")

    # Load all necessary data
    with open(os.path.join(previous_run_path, "transforms.json"), 'r') as f:
        previous_data = json.load(f)
    with open(full_dataset_json_path, 'r') as f:
        full_data = json.load(f)
    with open(recommendations_path, 'r') as f:
        recommended_paths = {line.strip() for line in f}

    print(f"Loaded {len(recommended_paths)} new images to add.")

    # Create a lookup map for the full dataset for quick access
    full_frames_map = {frame['file_path']: frame for frame in full_data['frames']}

    # Start with the frames from the previous run
    new_frames = previous_data['frames']
    
    # Add the new recommended frames
    for path in recommended_paths:
        if path in full_frames_map:
            new_frames.append(full_frames_map[path])
        else:
            print(f"Warning: Recommended path '{path}' not found in full dataset.")
            
    # Remove duplicates (just in case)
    # This is a bit inefficient but robust
    seen_paths = set()
    unique_new_frames = []
    for frame in new_frames:
        if frame['file_path'] not in seen_paths:
            unique_new_frames.append(frame)
            seen_paths.add(frame['file_path'])

    # Create the new augmented data structure
    augmented_data = {
        'camera_angle_x': full_data.get('camera_angle_x'),
        'frames': unique_new_frames
    }

    print(f"New dataset has {len(augmented_data['frames'])} frames.")

    # Write the new transforms.json for the augmented set
    output_json_path = os.path.join(new_run_dir, "transforms.json")
    with open(output_json_path, 'w') as f:
        json.dump(augmented_data, f, indent=4)
        
    print(f"Successfully created augmented dataset at: {output_json_path}")
    print("--- Augmentation complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment a dataset with recommended images.")
    parser.add_argument("--previous_run", required=True, help="Path to the completed run directory to augment from.")
    parser.add_argument("--new_run_name", required=True, help="Name for the new augmented experiment run (e.g., 'run_02_augmented').")
    parser.add_argument("--source", default="data/{dataset}/transforms_train.json",
                        help="Path template to the full transforms JSON.")
    
    args = parser.parse_args()

    dataset_name = args.previous_run.split(os.sep)[1]
    full_json_path = args.source.format(dataset=dataset_name)

    augment_dataset(args.previous_run, args.new_run_name, full_json_path)