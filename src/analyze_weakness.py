# src/2_analyze_weakness.py

import sys
import os
from tqdm import trange

# Note: We do not need pyngp at all, which simplifies things.
import numpy as np
import json
import subprocess
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2
from pathlib import Path

def analyze_weakness(run_path, full_dataset_json_path, mode):
    """
    Loads a pre-trained scout model and analyzes it by rendering novel views
    using the official Instant-NGP `scripts/run.py`.
    """
    print(f"--- Analyzing weakness for run: {run_path} ---")
    print(f"--- Mode: '{mode.upper()}' ---")
    
    analysis_dir = os.path.join(run_path, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    snapshot_path = os.path.abspath(os.path.join(run_path, "scout_model/scout.ingp"))
    official_script_path = os.path.abspath("./vendor/instant-ngp/scripts/run.py")
    scene_json_path = os.path.abspath(os.path.join(run_path, "transforms.json"))

    if not os.path.exists(snapshot_path):
        print(f"❌ ERROR: Snapshot file not found at '{snapshot_path}'. Cannot analyze.")
        exit()

    print(f"Step 2.1: Analyzing model '{snapshot_path}' using '{mode}' metric...")
    radius = 4.0
    n_test_views = 256
    view_positions_spherical = []
    metric_scores = []
    weak_view_positions_cartesian = []
    
    camera_path_file = os.path.abspath(os.path.join(analysis_dir, "temp_cam_path.json"))
    screenshot_dir = os.path.abspath(analysis_dir)

    # Load the original full dataset once to get the camera_angle_x value, which is required by the script
    with open(full_dataset_json_path, 'r') as f:
        full_data = json.load(f)
    camera_angle_x = full_data.get("camera_angle_x", 0.691111) # Use Lego default as a safe fallback

    print(f"Probing model from {n_test_views} different angles...")
    for i in trange(n_test_views):
        phi = np.arccos(1 - 2 * (i + 0.5) / n_test_views)
        theta = np.pi * (1 + 5**0.5) * i
        view_positions_spherical.append((theta, phi))
        cam_pos_cartesian = np.array([radius * np.cos(theta) * np.sin(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(phi)])
        
        forward = -cam_pos_cartesian / np.linalg.norm(cam_pos_cartesian)
        right = np.cross(np.array([0, 0, 1]), forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        
        cam_matrix = np.eye(4)
        cam_matrix[:3, 0] = right
        cam_matrix[:3, 1] = up
        cam_matrix[:3, 2] = forward
        cam_matrix[:3, 3] = cam_pos_cartesian
        
        # 1. Create the camera path JSON file in the format the script expects.
        #    THE FINAL FIX: We add a dummy 'file_path' to satisfy the script's output naming logic.
        camera_data = {
            "camera_angle_x": camera_angle_x,
            "frames": [{
                "transform_matrix": cam_matrix.tolist(),
                "file_path": f"render_{i:04d}.png" 
            }]
        }
        with open(camera_path_file, 'w') as f:
            json.dump(camera_data, f)
            
        # 2. Call the official script to render a screenshot.
        cmd_render = [
            "python", official_script_path,
            "--scene", scene_json_path,
            "--load_snapshot", snapshot_path,
            "--screenshot_transforms", camera_path_file,
            "--screenshot_dir", screenshot_dir,
            "--width", "400",
            "--height", "400",
            "--screenshot_spp", "8"
        ]
        subprocess.run(cmd_render, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 3. Load the output image. The official script now names them based on our dummy file_path.
        output_image_path = os.path.join(screenshot_dir, f"render_{i:04d}.png")
        rgba_image = cv2.imread(output_image_path, cv2.IMREAD_UNCHANGED)
        
        if rgba_image is None:
            print(f"Warning: Failed to read rendered image for view {i}. Skipping.")
            metric_scores.append(1.0 if mode == 'geometry' else 0)
            continue

        # --- ANALYSIS LOGIC ---
        if mode == 'geometry':
            opacity_map = rgba_image[:, :, 3] / 255.0
            mean_opacity = np.mean(opacity_map)
            metric_scores.append(1.0 - mean_opacity)
            if mean_opacity < 0.95:
                weak_view_positions_cartesian.append(cam_pos_cartesian)
        elif mode == 'detail':
            color_image = rgba_image[:, :, :3]
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            lap_variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            metric_scores.append(lap_variance)

    # --- DETAIL MODE THRESHOLDING ---
    if mode == 'detail':
        percentile_threshold = np.percentile(metric_scores, 20)
        print(f"Detail mode: Identifying views with Laplacian Variance < {percentile_threshold:.2f}")
        # Need to create a new list for weak views in detail mode
        detail_weak_views = []
        for i, score in enumerate(metric_scores):
            if score < percentile_threshold:
                # Re-calculate the position
                phi = np.arccos(1 - 2 * (i + 0.5) / n_test_views)
                theta = np.pi * (1 + 5**0.5) * i
                weak_pos = np.array([radius*np.cos(theta)*np.sin(phi), radius*np.sin(theta)*np.sin(phi), radius*np.cos(phi)])
                detail_weak_views.append(weak_pos)
        weak_view_positions_cartesian = detail_weak_views
        
        max_score = np.max(metric_scores) if metric_scores else 0
        metric_scores = max_score - np.array(metric_scores)
        
    print(f"Found {len(weak_view_positions_cartesian)} weak viewpoints.")
    print("Step 3: Generating uncertainty heatmap...")
    # ... (heatmap logic)
    thetas, phis = zip(*view_positions_spherical)
    points = np.array([thetas, phis]).T
    grid_theta, grid_phi = np.mgrid[0:2*np.pi:500j, 0:np.pi:250j]
    grid_uncertainty = griddata(points, metric_scores, (grid_theta, grid_phi), method='cubic', fill_value=np.nan)
    fig, ax = plt.subplots(figsize=(10, 5)); cax = ax.imshow(grid_uncertainty.T, extent=[360, 0, 180, 0], cmap='plasma', origin='upper')
    ax.set_title(f"Model Uncertainty Heatmap (Mode: {mode.upper()})"); ax.set_xlabel("Theta (Azimuth)"); ax.set_ylabel("Phi (Inclination)")
    fig.colorbar(cax, label="Uncertainty Score (Higher is Weaker)"); heatmap_path = os.path.join(analysis_dir, f"uncertainty_heatmap_{mode}.png")
    plt.savefig(heatmap_path, bbox_inches='tight'); plt.close(fig)
    print(f"Heatmap saved to {heatmap_path}")
    
    if not weak_view_positions_cartesian:
        print("No new images recommended at this threshold.")
        return
        
    print("Step 4: Finding best candidate images from full dataset...")
    # ... (recommendation logic)
    with open(full_dataset_json_path, 'r') as f: full_frames = json.load(f)['frames']
    with open(os.path.join(run_path, "transforms.json"), 'r') as f: current_frame_paths = {frame['file_path'] for frame in json.load(f)['frames']}
    candidate_frames = [f for f in full_frames if f['file_path'] not in current_frame_paths]
    best_suggestions = set()
    for weak_pos in weak_view_positions_cartesian:
        min_dist = float('inf'); best_candidate_path = None
        for frame in candidate_frames:
            cam_mat = np.array(frame['transform_matrix']); cam_pos = cam_mat[:3, 3]; dist = np.linalg.norm(weak_pos - cam_pos)
            if dist < min_dist: min_dist = dist; best_candidate_path = frame['file_path']
        if best_candidate_path: best_suggestions.add(best_candidate_path)
    output_rec_path = os.path.join(run_path, "recommended_images.txt")
    with open(output_rec_path, 'w') as f:
        for path in sorted(list(best_suggestions)): f.write(path + "\n")
    print(f"Saved {len(best_suggestions)} recommendations to {output_rec_path}")
    print("--- Weakness analysis complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a pre-trained NGP model for weaknesses.")
    parser.add_argument("--run_path", required=True)
    parser.add_argument("--mode", choices=['geometry', 'detail'], default='geometry')
    args = parser.parse_args()
    
    dataset_name = Path(args.run_path).parts[1]
    full_json_path = f"data/{dataset_name}/transforms_train.json"
    
    if not os.path.exists(args.run_path):
        print(f"Error: Run path does not exist at '{args.run_path}'")
    elif not os.path.exists(full_json_path):
        print(f"Error: Full dataset JSON not found at '{full_json_path}'")
    else:
        analyze_weakness(args.run_path, full_json_path, args.mode)