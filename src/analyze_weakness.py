# src/2_analyze_weakness.py

import sys
import os
from tqdm import trange
import numpy as np
import json
import subprocess
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2
from pathlib import Path

def find_executable():
    """Finds the correct path to the Instant-NGP executable."""
    exe_name = "instant-ngp.exe"
    paths_to_check = [f"./vendor/instant-ngp/build/Release/{exe_name}", f"./vendor/instant-ngp/build/{exe_name}"]
    for path in paths_to_check:
        if os.path.exists(path): return path
    linux_path = "./vendor/instant-ngp/build/testbed"
    if os.path.exists(linux_path): return linux_path
    return None

def look_at(camera_pos, target_pos=np.array([0,0,0]), world_up=np.array([0, 0, 1])):
    """Computes a correct look-at camera-to-world matrix using the COLMAP/OpenCV convention."""
    forward_vector = camera_pos - target_pos
    forward_vector /= np.linalg.norm(forward_vector)
    right_vector = np.cross(world_up, forward_vector)
    right_vector /= np.linalg.norm(right_vector)
    up_vector = np.cross(forward_vector, right_vector)
    cam_to_world = np.eye(4)
    cam_to_world[:3, 0] = right_vector; cam_to_world[:3, 1] = up_vector; cam_to_world[:3, 2] = forward_vector; cam_to_world[:3, 3] = camera_pos
    return cam_to_world

def analyze_weakness(run_path, full_dataset_json_path, mode):
    print(f"--- Analyzing weakness for run: {run_path} ---"); print(f"--- Mode: '{mode.upper()}' ---")
    
    analysis_dir = os.path.join(run_path, "analysis"); os.makedirs(analysis_dir, exist_ok=True)
    snapshot_path = os.path.abspath(os.path.join(run_path, "scout_model/scout.ingp"))
    scene_json_path = os.path.abspath(os.path.join(run_path, "transforms.json"))
    official_script_path = os.path.abspath("./vendor/instant-ngp/scripts/run.py")

    if not os.path.exists(snapshot_path): print("❌ ERROR: Snapshot file not found."); exit()

    print(f"Step 2.1: Analyzing model '{snapshot_path}' using '{mode}' metric...")
    with open(scene_json_path, 'r') as f: training_data = json.load(f)
    positions = [np.array(frame["transform_matrix"])[:3, 3] for frame in training_data["frames"]]
    radius = np.mean([np.linalg.norm(p) for p in positions]) * 1.5
    camera_angle_x = training_data.get("camera_angle_x", 0.691111)
    print(f"Using camera sphere radius: {radius:.2f} (assuming centered data)")

    n_test_views = 64
    view_positions_spherical, metric_scores, weak_view_positions_cartesian = [], [], []
    
    camera_path_file = os.path.abspath(os.path.join(analysis_dir, "temp_cam_path.json"))
    screenshot_dir = os.path.abspath(analysis_dir)

    for i in trange(n_test_views, desc="Probing Model Views"):
        phi = np.arccos(1-2*(i+0.5)/n_test_views); theta = np.pi*(1+5**0.5)*i
        view_positions_spherical.append((theta, phi))
        cam_pos_cartesian = np.array([radius*np.cos(theta)*np.sin(phi), radius*np.sin(theta)*np.sin(phi), radius*np.cos(phi)])
        
        cam_matrix = look_at(cam_pos_cartesian)
        
        camera_data = { "camera_angle_x": camera_angle_x, "frames": [{"transform_matrix": cam_matrix.tolist(), "file_path": "render.png"}] }
        with open(camera_path_file, 'w') as f: json.dump(camera_data, f)
            
        cmd_render = [ "python", official_script_path, "--scene", scene_json_path, "--load_snapshot", snapshot_path, "--screenshot_transforms", camera_path_file, "--screenshot_dir", screenshot_dir, "--width", "400", "--height", "400", "--screenshot_spp", "8"]
        subprocess.run(cmd_render, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        output_image_path = os.path.join(screenshot_dir, "render.png")
        rgba_image = cv2.imread(output_image_path, cv2.IMREAD_UNCHANGED)
        if rgba_image is None: continue

        # --- THE NEW, ROBUST ANALYSIS LOGIC ---
        if mode == 'geometry':
            opacity_map = rgba_image[:, :, 3] / 255.0
            
            # 1. Clarity Check: Is the background clear?
            # A good view should have a significant portion of empty space (low opacity pixels).
            clear_background_ratio = np.sum(opacity_map < 0.1) / opacity_map.size
            
            # 2. Solidity Check: Of the parts that aren't clear, are they solid?
            # We filter out the background to avoid unfairly penalizing views of small objects.
            object_pixels = opacity_map[opacity_map > 0.1]
            if object_pixels.size > 0:
                mean_object_solidity = np.mean(object_pixels)
            else:
                # If there are NO object pixels, it's an completely empty render.
                mean_object_solidity = 0.0

            # A view is weak if the background isn't clear OR if the object itself isn't solid.
            is_weak = False
            uncertainty_score = 1.0 - mean_object_solidity # Default score
            if clear_background_ratio < 0.2: # Less than 20% clear background means it's likely all fog
                is_weak = True
                uncertainty_score = 1.0 # Max uncertainty for foggy views
            elif mean_object_solidity < 0.95: # If the object that IS there is semi-transparent
                is_weak = True
            
            metric_scores.append(uncertainty_score)
            if is_weak:
                weak_view_positions_cartesian.append(cam_pos_cartesian)

        elif mode == 'detail':
            color_image = rgba_image[:, :, :3]; gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            lap_variance = cv2.Laplacian(gray_image, cv2.CV_64F).var(); metric_scores.append(lap_variance)

    # Full detail mode logic restored
    if mode == 'detail':
        percentile_threshold = np.percentile(metric_scores, 20)
        print(f"Detail mode: Identifying views with Laplacian Variance < {percentile_threshold:.2f}")
        detail_weak_views = []
        for i, score in enumerate(metric_scores):
            if score < percentile_threshold:
                phi = np.arccos(1 - 2 * (i + 0.5) / n_test_views); theta = np.pi * (1 + 5**0.5) * i
                weak_pos = np.array([radius*np.cos(theta)*np.sin(phi), radius*np.sin(theta)*np.sin(phi), radius*np.cos(phi)])
                detail_weak_views.append(weak_pos)
        weak_view_positions_cartesian = detail_weak_views
        max_score = np.max(metric_scores) if metric_scores else 0
        metric_scores = max_score - np.array(metric_scores)
        
    print(f"Found {len(weak_view_positions_cartesian)} weak viewpoints.")
    print("Step 3: Generating uncertainty heatmap...")
    # Full heatmap logic restored
    thetas, phis = zip(*view_positions_spherical)
    points = np.array([thetas, phis]).T
    grid_theta, grid_phi = np.mgrid[-np.pi:np.pi:500j, 0:np.pi:250j]
    grid_uncertainty = griddata(points, metric_scores, (grid_theta, grid_phi), method='cubic', fill_value=np.nan)
    fig, ax = plt.subplots(figsize=(10, 5)); cax = ax.imshow(grid_uncertainty.T, extent=[360, 0, 180, 0], cmap='plasma', origin='upper', aspect='auto')
    ax.set_title(f"Model Uncertainty Heatmap (Mode: {mode.upper()})"); ax.set_xlabel("Theta (Azimuth)"); ax.set_ylabel("Phi (Inclination)")
    fig.colorbar(cax, label="Uncertainty Score (Higher is Weaker)"); heatmap_path = os.path.join(analysis_dir, f"uncertainty_heatmap_{mode}.png")
    plt.savefig(heatmap_path, bbox_inches='tight'); plt.close(fig)
    print(f"Heatmap saved to {heatmap_path}")
    
    if not weak_view_positions_cartesian: print("No new images recommended."); return
    print("Step 4: Finding best candidate images from full dataset...")
    
    with open(full_dataset_json_path, 'r') as f: full_frames = json.load(f)['frames']
    candidate_frames = [f for f in full_frames] # We don't need to filter by current frames
    
    source_base_dir = Path(full_dataset_json_path).parent
    best_suggestions = set()
    for weak_pos in weak_view_positions_cartesian:
        min_dist = float('inf'); best_candidate_path = None
        for frame in candidate_frames:
            cam_mat = np.array(frame['transform_matrix']); cam_pos = cam_mat[:3, 3]; dist = np.linalg.norm(np.array(weak_pos) - cam_pos)
            if dist < min_dist:
                min_dist = dist
                best_candidate_path = frame['file_path']
        
        if best_candidate_path:
            # --- THE ELEGANT FIX IS HERE ---
            # Convert the found relative path to an absolute path before adding.
            absolute_path = str((source_base_dir / Path(best_candidate_path)).resolve())
            best_suggestions.add(absolute_path)
            
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





#THIS DIAGNOSTIC VERSION SUCCESSFULLY RENDERS 8 VIEWS CENTERED ON THE ORIGIN WITH FIXED CAMERA MATRICES
# # src/2_analyze_weakness.py (FINAL DIAGNOSTIC VERSION)

# import sys
# import os
# from tqdm import trange
# import numpy as np
# import json
# import subprocess
# import argparse
# from pathlib import Path
# import cv2 # Make sure cv2 is imported if you want to inspect images

# def find_executable():
#     """Finds the correct path to the Instant-NGP executable."""
#     exe_name = "instant-ngp.exe"
#     paths_to_check = [f"./vendor/instant-ngp/build/Release/{exe_name}", f"./vendor/instant-ngp/build/{exe_name}"]
#     for path in paths_to_check:
#         if os.path.exists(path): return path
#     linux_path = "./vendor/instant-ngp/build/testbed"
#     if os.path.exists(linux_path): return linux_path
#     return None

# def look_at(camera_pos, target_pos=np.array([0,0,0]), world_up=np.array([0, 0, 1])):
#     """
#     Computes a correct look-at camera-to-world matrix using the COLMAP/OpenCV convention.
#     """
#     forward_vector = camera_pos - target_pos
#     forward_vector /= np.linalg.norm(forward_vector)
    
#     right_vector = np.cross(world_up, forward_vector)
#     right_vector /= np.linalg.norm(right_vector)
    
#     up_vector = np.cross(forward_vector, right_vector) # Corrected cross product order

#     cam_to_world = np.eye(4)
#     cam_to_world[:3, 0] = right_vector
#     cam_to_world[:3, 1] = up_vector # Standard Up, not inverted
#     cam_to_world[:3, 2] = forward_vector
#     cam_to_world[:3, 3] = camera_pos
    
#     return cam_to_world

# def analyze_weakness(run_path, full_dataset_json_path, mode):
#     """
#     Loads a scout model and runs a diagnostic test with increased radius and debug prints.
#     """
#     print(f"--- Running DIAGNOSTIC for: {run_path} ---"); print(f"--- Mode: '{mode.upper()}' ---")
    
#     analysis_dir = os.path.join(run_path, "analysis"); os.makedirs(analysis_dir, exist_ok=True)
#     snapshot_path = os.path.abspath(os.path.join(run_path, "scout_model/scout.ingp"))
#     scene_json_path = os.path.abspath(os.path.join(run_path, "transforms.json"))
#     official_script_path = os.path.abspath("./vendor/instant-ngp/scripts/run.py")

#     if not os.path.exists(snapshot_path): print("❌ ERROR: Snapshot file not found."); exit()

#     print(f"Step 2.1: Analyzing model '{snapshot_path}'...")
#     with open(scene_json_path, 'r') as f: training_data = json.load(f)
#     positions = [np.array(frame["transform_matrix"])[:3, 3] for frame in training_data["frames"]]
    
#     # --- MODIFIED FOR DIAGNOSTIC ---
#     radius = np.mean([np.linalg.norm(p) for p in positions]) * 2.0 # Increased radius
#     camera_angle_x = training_data.get("camera_angle_x", 0.691111)
#     print(f"Using INCREASED camera sphere radius: {radius:.2f}")

#     n_test_views = 8 # Reduced number of views for a fast test
    
#     camera_path_file = os.path.abspath(os.path.join(analysis_dir, "temp_cam_path.json"))
#     screenshot_dir = os.path.abspath(analysis_dir)

#     print(f"Probing model from {n_test_views} different angles with debug output...")
#     for i in range(n_test_views):
#         phi = np.arccos(1-2*(i+0.5)/n_test_views); theta = np.pi*(1+5**0.5)*i
#         cam_pos_cartesian = np.array([radius*np.cos(theta)*np.sin(phi), radius*np.sin(theta)*np.sin(phi), radius*np.cos(phi)])
        
#         # We will use the final, correct look_at function
#         cam_matrix = look_at(cam_pos_cartesian)
        
#         # --- DEBUG PRINTS ---
#         print("\n" + "="*50)
#         print(f"      DEBUGGING VIEW #{i}")
#         print("="*50)
#         print(f"Calculated Camera Position: {np.round(cam_pos_cartesian, 4)}")
#         print("Generated Camera-to-World Transform Matrix:")
#         np.set_printoptions(suppress=True, precision=4)
#         print(cam_matrix)
#         print("="*50)
        
#         # Save each render to a unique file for inspection
#         output_filename = f"render_debug_{i:04d}.png"
        
#         camera_data = {
#             "camera_angle_x": camera_angle_x,
#             "frames": [{"transform_matrix": cam_matrix.tolist(), "file_path": output_filename}]
#         }
#         with open(camera_path_file, 'w') as f: json.dump(camera_data, f)
            
#         cmd_render = [ "python", official_script_path, "--scene", scene_json_path, "--load_snapshot", snapshot_path, "--screenshot_transforms", camera_path_file, "--screenshot_dir", screenshot_dir, "--width", "400", "--height", "400", "--screenshot_spp", "8"]
        
#         try:
#             subprocess.run(cmd_render, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             print(f"✅ View {i} rendered successfully to {output_filename}")
#         except subprocess.CalledProcessError as e:
#             print(f"❌ ERROR: Rendering failed for view {i}. Check camera matrix above for issues.")
#             print(e)
            
#     print("\n--- Diagnostic complete. Please review the 8 debug blocks and the generated images. ---")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run a diagnostic test on the analysis script.")
#     parser.add_argument("--run_path", required=True)
#     parser.add_argument("--mode", choices=['geometry', 'detail'], default='geometry') # Mode is unused but kept for compatibility
#     args = parser.parse_args()
    
#     dataset_name = Path(args.run_path).parts[1]
#     full_json_path = f"data/{dataset_name}/transforms_train.json"
#     analyze_weakness(args.run_path, full_json_path, args.mode)