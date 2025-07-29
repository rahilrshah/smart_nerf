# src/2_analyze_weakness.py

import pyngp as ngp
import numpy as np
import json
import os
import subprocess
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2

def analyze_weakness(run_path, full_dataset_json_path, mode, n_steps):
    """
    Analyzes a model for weaknesses using one of two modes:
    1. 'geometry': Finds holes and floaters using mean opacity (for initial runs).
    2. 'detail': Finds blurry/low-detail areas using Laplacian variance (for later runs).
    """
    print(f"--- Analyzing weakness for run: {run_path} ---")
    print(f"--- Mode: '{mode.upper()}', Training Steps: {n_steps} ---")

    analysis_dir = os.path.join(run_path, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # --- Part A: Train the Scout Model ---
    scene_json_path = os.path.join(run_path, "transforms.json")
    snapshot_dir = os.path.join(run_path, "scout_model")
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_path = os.path.join(snapshot_dir, "scout.ingp")

    print(f"Step 1: Training scout model for {n_steps} steps...")
    subprocess.run([
        "./vendor/instant-ngp/build/testbed",
        "--scene", scene_json_path,
        "--n_steps", str(n_steps),
        "--save_snapshot", snapshot_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Scout model saved to {snapshot_path}")

    # --- Part B: Analyze Model based on selected mode ---
    print(f"Step 2: Analyzing model using '{mode}' metric...")
    testbed = ngp.Testbed()
    testbed.load_snapshot(snapshot_path)

    radius = 4.0
    n_test_views = 256

    view_positions_spherical = []
    metric_scores = []
    weak_view_positions_cartesian = []

    print(f"Probing model from {n_test_views} different angles...")
    for i in range(n_test_views):
        # Use a Fibonacci lattice to get evenly distributed points on a sphere
        phi = np.arccos(1 - 2 * (i + 0.5) / n_test_views)
        theta = np.pi * (1 + 5**0.5) * i

        # Store spherical coordinates for heatmap
        view_positions_spherical.append((theta, phi))

        # Create camera-to-world matrix looking from this point to the origin
        cam_pos_cartesian = np.array([
            radius * np.cos(theta) * np.sin(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(phi)
        ])
        forward = -cam_pos_cartesian / np.linalg.norm(cam_pos_cartesian)
        right = np.cross(np.array([0, 0, 1]), forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)

        cam_matrix = np.eye(4)
        cam_matrix[:3, 0] = right
        cam_matrix[:3, 1] = up
        cam_matrix[:3, 2] = forward
        cam_matrix[:3, 3] = cam_pos_cartesian
        testbed.set_camera_to_world(cam_matrix)

        rgba_image = testbed.render(400, 400, 4, True)

        # --- Conditional Metric Calculation ---
        if mode == 'geometry':
            opacity_map = rgba_image[:, :, 3]
            mean_opacity = np.mean(opacity_map)
            metric_scores.append(1.0 - mean_opacity)  # Uncertainty = 1 - opacity
            if mean_opacity < 0.95:
                weak_view_positions_cartesian.append(cam_pos_cartesian)

        elif mode == 'detail':
            color_image = (rgba_image[:, :, :3] * 255).astype(np.uint8)
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
            lap_variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            metric_scores.append(lap_variance)

    # --- Conditional Thresholding for 'detail' mode ---
    if mode == 'detail':
        # "Weak" is relative, so flag the bottom 20% of views by sharpness
        percentile_threshold = np.percentile(metric_scores, 20)
        print(f"Detail mode: Identifying views with Laplacian Variance < {percentile_threshold:.2f}")
        for i, score in enumerate(metric_scores):
            if score < percentile_threshold:
                theta, phi = view_positions_spherical[i]
                weak_pos = np.array([radius * np.cos(theta)*np.sin(phi), radius * np.sin(theta)*np.sin(phi), radius*np.cos(phi)])
                weak_view_positions_cartesian.append(weak_pos)
        # Invert scores for heatmap: high value should mean "bad" (low variance)
        max_score = np.max(metric_scores) if metric_scores else 0
        metric_scores = max_score - np.array(metric_scores)

    print(f"Found {len(weak_view_positions_cartesian)} weak viewpoints.")

    # --- Part B.2: Generate and Save Uncertainty Heatmap ---
    print("Step 3: Generating uncertainty heatmap...")
    thetas, phis = zip(*view_positions_spherical)
    points = np.array([thetas, phis]).T
    grid_theta, grid_phi = np.mgrid[0:2*np.pi:500j, 0:np.pi:250j]
    grid_uncertainty = griddata(points, metric_scores, (grid_theta, grid_phi), method='cubic', fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(grid_uncertainty.T, extent=[360, 0, 180, 0], cmap='plasma', origin='upper')
    ax.set_title(f"Model Uncertainty Heatmap (Mode: {mode.upper()})")
    ax.set_xlabel("Theta (Azimuth)")
    ax.set_ylabel("Phi (Inclination)")
    fig.colorbar(cax, label="Uncertainty Score (Higher is Weaker)")
    heatmap_path = os.path.join(analysis_dir, f"uncertainty_heatmap_{mode}.png")
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory
    print(f"Heatmap saved to {heatmap_path}")

    if not weak_view_positions_cartesian:
        print("No new images recommended at this threshold.")
        return

    # --- Part C: Recommend Best Images to Fix Weaknesses ---
    print("Step 4: Finding best candidate images from full dataset...")
    with open(full_dataset_json_path, 'r') as f:
        full_frames = json.load(f)['frames']
    with open(scene_json_path, 'r') as f:
        current_frame_paths = {frame['file_path'] for frame in json.load(f)['frames']}
    
    candidate_frames = [f for f in full_frames if f['file_path'] not in current_frame_paths]
    best_suggestions = set()

    for weak_pos in weak_view_positions_cartesian:
        min_dist = float('inf')
        best_candidate_path = None
        for frame in candidate_frames:
            cam_mat = np.array(frame['transform_matrix'])
            cam_pos = cam_mat[:3, 3]
            dist = np.linalg.norm(weak_pos - cam_pos)
            if dist < min_dist:
                min_dist = dist
                best_candidate_path = frame['file_path']
        if best_candidate_path:
            best_suggestions.add(best_candidate_path)

    output_rec_path = os.path.join(run_path, "recommended_images.txt")
    with open(output_rec_path, 'w') as f:
        for path in sorted(list(best_suggestions)):
            f.write(path + "\n")
            
    print(f"Saved {len(best_suggestions)} recommendations to {output_rec_path}")
    print("--- Weakness analysis complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze an NGP model for weaknesses in geometry or detail.")
    parser.add_argument("--run_path", required=True, help="Path to the experiment run directory (e.g., 'experiments/lego/run_01').")
    parser.add_argument("--source", default="data/{dataset}/transforms_train.json", help="Path template to the full transforms JSON.")
    parser.add_argument("--mode", choices=['geometry', 'detail'], default='geometry', help="Analysis mode: 'geometry' for holes, 'detail' for blurriness.")
    parser.add_argument("--n_steps", type=int, default=3500, help="Number of training steps for the scout model.")
    
    args = parser.parse_args()
    
    try:
        dataset_name = args.run_path.split(os.sep)[1]
    except IndexError:
        print(f"Error: Could not determine dataset name from --run_path '{args.run_path}'. Expected format 'experiments/DATASET_NAME/RUN_NAME'.")
        exit()

    full_json_path = args.source.format(dataset=dataset_name)

    if not os.path.exists(args.run_path):
        print(f"Error: Run path does not exist at '{args.run_path}'")
    elif not os.path.exists(full_json_path):
        print(f"Error: Full dataset JSON not found at '{full_json_path}'")
    else:
        analyze_weakness(args.run_path, full_json_path, args.mode, args.n_steps)