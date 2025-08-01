# src/run_neuralangelo.py

import argparse
import subprocess
import os
import json
import numpy as np

def prepare_data_for_neuralangelo(optimized_json_path, output_data_dir):
    """
    Converts instant-ngp's transforms.json into the data format
    expected by Neuralangelo (images, masks, poses).
    """
    print(f"--- Preparing data for Neuralangelo in {output_data_dir} ---")
    
    # Create the required subdirectories
    img_dir = os.path.join(output_data_dir, "images")
    pose_dir = os.path.join(output_data_dir, "poses")
    # Neuralangelo works better with masks. For blender data, we can create trivial ones.
    mask_dir = os.path.join(output_data_dir, "masks")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    with open(optimized_json_path, 'r') as f:
        data = json.load(f)
    
    # Base directory of the images (e.g., data/lego)
    base_dir = os.path.dirname(os.path.dirname(optimized_json_path))

    for frame in data['frames']:
        # Copy image file
        img_filename = os.path.basename(frame['file_path'])
        src_img_path = os.path.join(base_dir, frame['file_path'])
        dst_img_path = os.path.join(img_dir, img_filename)
        # Using a hard link is faster than copying if possible
        if not os.path.exists(dst_img_path):
             os.link(src_img_path, dst_img_path)

        # Write camera pose matrix to a .txt file
        pose_filename = os.path.splitext(img_filename)[0] + ".txt"
        pose = np.array(frame['transform_matrix'])
        np.savetxt(os.path.join(pose_dir, pose_filename), pose)

        # Create a trivial all-white mask (assuming object is the whole image)
        # NOTE: For real-world data, providing real masks is MUCH better.
        mask_filename = img_filename
        # You would typically load the image to get its size here. We assume 800x800 for Lego.
        mask = np.ones((800, 800), dtype=np.uint8) * 255
        import imageio
        imageio.imwrite(os.path.join(mask_dir, mask_filename), mask)

    print("--- Data preparation complete ---")


def run_neuralangelo_pipeline(optimized_json_path, output_base_dir):
    """
    Orchestrates the full Neuralangelo pipeline: data prep, training, meshing.
    """
    dataset_name = optimized_json_path.split(os.sep)[1]
    neuralangelo_data_dir = os.path.join(output_base_dir, dataset_name, "neuralangelo_data")
    neuralangelo_output_dir = os.path.join(output_base_dir, dataset_name, "neuralangelo_output")

    # Step 1: Convert our data to the format Neuralangelo needs.
    prepare_data_for_neuralangelo(optimized_json_path, neuralangelo_data_dir)

    # Step 2: Launch Neuralangelo training.
    # NOTE: This command is a TEMPLATE. You MUST adapt it based on the official
    # Neuralangelo documentation and configuration files.
    print("\n--- Starting Neuralangelo Training ---")
    training_cmd = [
        "python", "projects/neuralangelo/train.py",
        f"--data_dir={neuralangelo_data_dir}",
        f"--log_dir={neuralangelo_output_dir}",
        # Add other necessary arguments like --config, etc.
        # Example for object-centric:
        "--config=projects/neuralangelo/configs/custom/lego.yaml"
    ]
    # IMPORTANT: You need to `cd` into the vendor directory or adjust paths
    subprocess.run(training_cmd, cwd="vendor/neuralangelo", check=True)
    print("--- Training complete ---")

    # Step 3: Launch Mesh Extraction
    # NOTE: This also a TEMPLATE. You need to find the trained config file.
    print("\n--- Starting Neuralangelo Mesh Extraction ---")
    trained_config = glob.glob(f"{neuralangelo_output_dir}/**/config.yaml", recursive=True)[0]
    meshing_cmd = [
        "python", "projects/neuralangelo/extract_mesh.py",
        f"--config={trained_config}"
        # Add other arguments as needed
    ]
    subprocess.run(meshing_cmd, cwd="vendor/neuralangelo", check=True)
    print("--- Meshing complete. Find your .obj file in the experiment output folder. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Neuralangelo pipeline on an optimized dataset.")
    parser.add_argument("--optimized_json", required=True, help="Path to the final, optimized transforms.json file.")
    parser.add_argument("--output_base", default="results", help="Base directory to save final outputs.")
    
    args = parser.parse_args()

    if not os.path.exists(args.optimized_json):
        print(f"Error: Optimized JSON file not found at {args.optimized_json}")
    else:
        run_neuralangelo_pipeline(args.optimized_json, args.output_base)