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

def analyze_weakness(run_path, full_dataset_json_path, mode, max_recs):
    """
    Loads a scout model, analyzes it with the robust metrics we developed, and
    recommends the top N best new images.
    """
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
    print(f"Using camera sphere radius: {radius:.2f}")

    n_test_views = 64 # Keep analysis fast but thorough
    view_positions_spherical, metric_scores, weak_view_info = [], [], []
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

        # --- THE CORRECT, ROBUST GEOMETRY ANALYSIS LOGIC ---
        if mode == 'geometry':
            opacity_map = rgba_image[:, :, 3] / 255.0
            
            # 1. Clarity Check: A good view must have some clear background.
            clear_background_ratio = np.sum(opacity_map < 0.1) / opacity_map.size
            
            # 2. Solidity Check: Of the parts that are not background, they must be solid.
            object_pixels = opacity_map[opacity_map > 0.1]
            mean_object_solidity = np.mean(object_pixels) if object_pixels.size > 0 else 0.0

            is_weak = False
            uncertainty_score = 1.0 - mean_object_solidity
            if clear_background_ratio < 0.2: # Less than 20% clear background => foggy mess
                is_weak = True
                uncertainty_score = 1.0 # Maximum uncertainty
            elif mean_object_solidity < 0.95: # Object is translucent/ghostly
                is_weak = True
            
            metric_scores.append(uncertainty_score)
            if is_weak:
                weak_view_info.append({"pos": cam_pos_cartesian, "score": uncertainty_score})

        elif mode == 'detail':
            color_image = rgba_image[:, :, :3]; gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            lap_variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            metric_scores.append(lap_variance)
            weak_view_info.append({"pos": cam_pos_cartesian, "score": lap_variance})
    
    # --- FULL LOGIC RESTORED ---
    print(f"Found {len(weak_view_info)} potential weak viewpoints.")
    
    if mode == 'detail':
        percentile_threshold = np.percentile([v["score"] for v in weak_view_info], 20)
        weak_view_info = [v for v in weak_view_info if v["score"] < percentile_threshold]
        max_score = np.max(metric_scores) if metric_scores else 0
        metric_scores = max_score - np.array(metric_scores)
        
    print(f"Identified {len(weak_view_info)} actionable weak viewpoints.")
    
    if not weak_view_info:
        print("No new images recommended at this threshold.")
        # Create an empty recommendations file to signal convergence
        output_rec_path = os.path.join(run_path, "recommended_images.txt")
        open(output_rec_path, 'w').close()
        return

    # --- FULL HEATMAP LOGIC RESTORED ---
    print("Step 3: Generating uncertainty heatmap...")
    thetas, phis = zip(*view_positions_spherical)
    points = np.array([thetas, phis]).T
    grid_theta, grid_phi = np.mgrid[-np.pi:np.pi:500j, 0:np.pi:250j]
    grid_uncertainty = griddata(points, metric_scores, (grid_theta, grid_phi), method='cubic', fill_value=np.nan)
    fig, ax = plt.subplots(figsize=(10, 5)); cax = ax.imshow(grid_uncertainty.T, extent=[360, 0, 180, 0], cmap='plasma', origin='upper', aspect='auto')
    ax.set_title(f"Model Uncertainty Heatmap (Mode: {mode.upper()})"); ax.set_xlabel("Theta (Azimuth)"); ax.set_ylabel("Phi (Inclination)")
    fig.colorbar(cax, label="Uncertainty Score (Higher is Weaker)"); heatmap_path = os.path.join(analysis_dir, f"uncertainty_heatmap_{mode}.png")
    plt.savefig(heatmap_path, bbox_inches='tight'); plt.close(fig)
    print(f"Heatmap saved to {heatmap_path}")

    # --- FULL RECOMMENDATION LOGIC RESTORED ---
    print("Step 4: Finding best candidate images from full dataset...")
    
    with open(full_dataset_json_path, 'r') as f: full_frames = json.load(f)['frames']
    source_base_dir = Path(full_dataset_json_path).parent

    # --- THE FINAL, DEFINITIVE FIX IS HERE ---
    # 1. Get the set of absolute paths that are ALREADY in our current training set.
    with open(scene_json_path, 'r') as f:
        current_frames_data = json.load(f)['frames']
    current_absolute_paths = {frame['file_path'] for frame in current_frames_data}

    # 2. Create the list of available candidate frames by filtering out used images.
    candidate_frames = []
    for frame in full_frames:
        # Convert the original relative path to an absolute path for comparison
        abs_path = str((source_base_dir / Path(frame['file_path'])).resolve())
        if abs_path not in current_absolute_paths:
            candidate_frames.append(frame)

    print(f"  - Searching for best matches among {len(candidate_frames)} available images.")

    weak_view_info.sort(key=lambda x: x["score"], reverse=True)
    best_suggestions = set()
    
    # 3. Now, we search for the closest match only within the pool of UNUSED images.
    for weak_view in weak_view_info:
        weak_pos = weak_view["pos"]
        min_dist = float('inf'); best_candidate_path = None
        for frame in candidate_frames:
            cam_pos = np.array(frame['transform_matrix'])[:3, 3]
            dist = np.linalg.norm(weak_pos - cam_pos)
            if dist < min_dist:
                min_dist = dist
                best_candidate_path = frame['file_path']
        
        if best_candidate_path:
            absolute_path = str((source_base_dir / Path(best_candidate_path)).resolve())
            best_suggestions.add(absolute_path)

    final_recommendations = sorted(list(best_suggestions))
    
    if max_recs > 0 and len(final_recommendations) > max_recs:
        print(f"Limiting {len(final_recommendations)} recommendations to a maximum of {max_recs}.")
        final_recommendations = final_recommendations[:max_recs]
            
    output_rec_path = os.path.join(run_path, "recommended_images.txt")
    with open(output_rec_path, 'w') as f:
        for path in final_recommendations: f.write(path + "\n")
    print(f"Saved {len(final_recommendations)} recommendations to {output_rec_path}")
    print("--- Weakness analysis complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a pre-trained NGP model for weaknesses.")
    parser.add_argument("--run_path", required=True)
    parser.add_argument("--mode", choices=['geometry', 'detail'], default='geometry')
    parser.add_argument("--max_recommendations", type=int, default=10, help="Maximum number of new images to recommend per iteration. Set to 0 for unlimited.")
    args = parser.parse_args()
    
    dataset_name = Path(args.run_path).parts[1]
    full_json_path = f"data/{dataset_name}/transforms_train.json"
    analyze_weakness(args.run_path, full_json_path, args.mode, args.max_recommendations)




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