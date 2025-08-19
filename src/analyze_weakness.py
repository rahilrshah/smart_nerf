# src/analyze_weakness_realtime.py
# Modified weakness analysis with real-time capture integration and adaptive sampling

import sys
import os
from tqdm import trange, tqdm
import numpy as np
import json
import subprocess
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2
from pathlib import Path
import concurrent.futures
import threading
import time
from typing import List, Dict, Tuple, Optional
import tempfile
import shutil

# Import the new real-time capture system
try:
    from camera_position_controller import (
        RealTimeCaptureSystem, 
        AdaptiveViewpointGenerator,
        integrate_with_weakness_analysis
    )
    REALTIME_AVAILABLE = True
except ImportError:
    print("Real-time capture system not available. Running in analysis-only mode.")
    REALTIME_AVAILABLE = False

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
    cam_to_world[:3, 0] = right_vector
    cam_to_world[:3, 1] = up_vector
    cam_to_world[:3, 2] = forward_vector
    cam_to_world[:3, 3] = camera_pos
    return cam_to_world

class ViewpointRenderer:
    """Handles rendering individual viewpoints with thread safety."""
    
    def __init__(self, snapshot_path: str, scene_json_path: str, official_script_path: str, 
                 camera_angle_x: float, output_dir: str):
        self.snapshot_path = snapshot_path
        self.scene_json_path = scene_json_path
        self.official_script_path = official_script_path
        self.camera_angle_x = camera_angle_x
        self.output_dir = output_dir
        self._lock = threading.Lock()
        
    def render_viewpoint(self, view_id: int, cam_matrix: np.ndarray) -> Optional[str]:
        """Render a single viewpoint and return the output image path."""
        try:
            # Create unique temporary files for this thread
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_json:
                camera_data = {
                    "camera_angle_x": self.camera_angle_x,
                    "frames": [{"transform_matrix": cam_matrix.tolist(), "file_path": f"render_{view_id:04d}.png"}]
                }
                json.dump(camera_data, temp_json)
                temp_json_path = temp_json.name
            
            output_filename = f"render_{view_id:04d}.png"
            output_path = os.path.join(self.output_dir, output_filename)
            
            cmd_render = [
                "python", self.official_script_path,
                "--scene", self.scene_json_path,
                "--load_snapshot", self.snapshot_path,
                "--screenshot_transforms", temp_json_path,
                "--screenshot_dir", self.output_dir,
                "--width", "400", "--height", "400",
                "--screenshot_spp", "8"
            ]
            
            result = subprocess.run(cmd_render, check=True, capture_output=True, text=True)
            
            # Clean up temp file
            os.unlink(temp_json_path)
            
            # Verify output exists
            if os.path.exists(output_path):
                return output_path
            else:
                print(f"Warning: Expected output {output_path} not found")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"Error rendering view {view_id}: {e}")
            if 'temp_json_path' in locals():
                os.unlink(temp_json_path)
            return None
        except Exception as e:
            print(f"Unexpected error rendering view {view_id}: {e}")
            return None

class WeaknessAnalyzer:
    """Analyzes rendered viewpoints for weaknesses with sophisticated scoring."""
    
    def __init__(self, mode: str = 'geometry'):
        self.mode = mode
        
    def calculate_geometry_score(self, rgba_image: np.ndarray) -> Dict:
        """Calculate geometry-based weakness score with detailed metrics."""
        opacity_map = rgba_image[:, :, 3] / 255.0
        
        # 1. Background clarity analysis
        clear_background_ratio = np.sum(opacity_map < 0.1) / opacity_map.size
        
        # 2. Object solidity analysis
        object_pixels = opacity_map[opacity_map > 0.1]
        mean_object_solidity = np.mean(object_pixels) if object_pixels.size > 0 else 0.0
        
        # 3. Edge consistency (new metric)
        edges = cv2.Canny((opacity_map * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 4. Opacity variance (new metric)
        opacity_variance = np.var(opacity_map)
        
        # 5. Fragmentation analysis (new metric)
        binary_mask = (opacity_map > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fragmentation = len(contours) / max(1, np.sum(binary_mask > 0) / 1000)  # Normalize by object size
        
        # Composite scoring with weights
        weights = {
            'background_clarity': 0.3,
            'object_solidity': 0.3,
            'edge_consistency': 0.2,
            'opacity_variance': 0.1,
            'fragmentation': 0.1
        }
        
        # Convert metrics to weakness scores (higher = weaker)
        weakness_components = {
            'background_clarity': max(0, 0.2 - clear_background_ratio) / 0.2,  # Bad if < 20% clear
            'object_solidity': max(0, 0.95 - mean_object_solidity) / 0.95,     # Bad if < 95% solid
            'edge_consistency': max(0, 0.05 - edge_density) / 0.05,            # Bad if too few edges
            'opacity_variance': min(1.0, opacity_variance * 2),                # Bad if high variance
            'fragmentation': min(1.0, fragmentation / 2)                       # Bad if fragmented
        }
        
        # Calculate weighted composite score
        composite_score = sum(weights[k] * weakness_components[k] for k in weights.keys())
        
        return {
            'score': composite_score,
            'components': weakness_components,
            'metrics': {
                'clear_background_ratio': clear_background_ratio,
                'mean_object_solidity': mean_object_solidity,
                'edge_density': edge_density,
                'opacity_variance': opacity_variance,
                'fragmentation': fragmentation
            }
        }
    
    def calculate_detail_score(self, rgba_image: np.ndarray) -> Dict:
        """Calculate detail-based weakness score with multiple blur metrics."""
        color_image = rgba_image[:, :, :3]
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # 1. Laplacian variance (original metric)
        lap_variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        
        # 2. Sobel gradient magnitude
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2).mean()
        
        # 3. High-frequency content analysis
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Focus on high-frequency components (outer regions)
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 > radius**2
        high_freq_energy = np.mean(magnitude_spectrum[mask])
        
        # 4. Texture analysis using Local Binary Patterns
        def calculate_lbp_variance(image, radius=3, n_points=24):
            """Calculate variance of Local Binary Pattern histogram."""
            from scipy import ndimage
            # Simplified LBP calculation
            neighbors = []
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                dy, dx = radius * np.sin(angle), radius * np.cos(angle)
                neighbor = ndimage.shift(image, [dy, dx], mode='constant')
                neighbors.append(neighbor > image)
            
            lbp = sum(neighbors[i] * (2**i) for i in range(len(neighbors)))
            hist, _ = np.histogram(lbp.flatten(), bins=n_points+2)
            return np.var(hist)
        
        texture_variance = calculate_lbp_variance(gray_image)
        
        # Normalize and combine metrics
        weights = {
            'laplacian': 0.4,
            'sobel': 0.3,
            'high_freq': 0.2,
            'texture': 0.1
        }
        
        # Convert to weakness scores (invert high-quality indicators)
        max_lap = 1000  # Typical good laplacian variance
        max_sobel = 100  # Typical good sobel magnitude
        max_hf = 10000  # Typical good high-freq energy
        max_texture = 1000  # Typical good texture variance
        
        weakness_components = {
            'laplacian': max(0, 1 - lap_variance / max_lap),
            'sobel': max(0, 1 - sobel_magnitude / max_sobel),
            'high_freq': max(0, 1 - high_freq_energy / max_hf),
            'texture': max(0, 1 - texture_variance / max_texture)
        }
        
        composite_score = sum(weights[k] * weakness_components[k] for k in weights.keys())
        
        return {
            'score': composite_score,
            'components': weakness_components,
            'metrics': {
                'laplacian_variance': lap_variance,
                'sobel_magnitude': sobel_magnitude,
                'high_freq_energy': high_freq_energy,
                'texture_variance': texture_variance
            }
        }
    
    def analyze_image(self, image_path: str) -> Optional[Dict]:
        """Analyze a single rendered image."""
        rgba_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if rgba_image is None:
            return None
            
        if self.mode == 'geometry':
            return self.calculate_geometry_score(rgba_image)
        else:  # detail mode
            return self.calculate_detail_score(rgba_image)

def generate_viewpoints_adaptive(n_views: int, radius: float, iteration: int = 0, 
                               capture_system: Optional['RealTimeCaptureSystem'] = None) -> List[Tuple[int, np.ndarray, Tuple[float, float]]]:
    """Generate adaptive viewpoints that change each iteration to avoid overfitting."""
    if REALTIME_AVAILABLE and capture_system:
        # Use adaptive viewpoint generation
        return capture_system.generate_adaptive_probe_viewpoints(n_views)
    else:
        # Fallback to modified fibonacci with iteration-based variation
        viewpoints = []
        
        # Add iteration-based offset to vary sampling pattern
        iteration_offset = iteration * 0.618034  # Golden ratio for good distribution
        
        for i in range(n_views):
            # Modified Fibonacci sphere with iteration-based perturbation
            phi_base = np.arccos(1 - 2 * (i + 0.5) / n_views)
            theta_base = np.pi * (1 + 5**0.5) * i
            
            # Add controlled randomization that changes with iteration
            np.random.seed(iteration * 1000 + i)  # Deterministic but iteration-dependent
            theta_perturb = np.random.normal(0, 0.1) * (iteration + 1) / 5
            phi_perturb = np.random.normal(0, 0.05) * (iteration + 1) / 5
            
            theta = theta_base + theta_perturb + iteration_offset
            phi = np.clip(phi_base + phi_perturb, 0.1, np.pi - 0.1)
            
            cam_pos_cartesian = np.array([
                radius * np.cos(theta) * np.sin(phi),
                radius * np.sin(theta) * np.sin(phi), 
                radius * np.cos(phi)
            ])
            
            cam_matrix = look_at(cam_pos_cartesian)
            viewpoints.append((i, cam_matrix, (theta, phi)))
        
        print(f"Generated {n_views} viewpoints using adaptive pattern for iteration {iteration}")
        return viewpoints

def render_viewpoints_parallel(renderer: ViewpointRenderer, viewpoints: List, max_workers: int = 4, batch_size: int = 8) -> List[Tuple[int, Optional[str], Tuple[float, float]]]:
    """Render viewpoints in parallel batches."""
    results = []
    
    # Process in batches to avoid overwhelming the system
    for batch_start in trange(0, len(viewpoints), batch_size, desc="Rendering Batches"):
        batch_end = min(batch_start + batch_size, len(viewpoints))
        batch_viewpoints = viewpoints[batch_start:batch_end]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch for rendering
            future_to_viewpoint = {
                executor.submit(renderer.render_viewpoint, view_id, cam_matrix): (view_id, spherical_pos)
                for view_id, cam_matrix, spherical_pos in batch_viewpoints
            }
            
            # Collect results with progress tracking
            batch_results = []
            for future in tqdm(concurrent.futures.as_completed(future_to_viewpoint), 
                             total=len(future_to_viewpoint), 
                             desc=f"Batch {batch_start//batch_size + 1}", 
                             leave=False):
                view_id, spherical_pos = future_to_viewpoint[future]
                try:
                    image_path = future.result()
                    batch_results.append((view_id, image_path, spherical_pos))
                except Exception as exc:
                    print(f'View {view_id} generated an exception: {exc}')
                    batch_results.append((view_id, None, spherical_pos))
            
            results.extend(batch_results)
        
        # Small delay between batches to prevent resource exhaustion
        if batch_end < len(viewpoints):
            time.sleep(0.5)
    
    return results

def analyze_weakness_realtime(run_path, full_dataset_json_path, mode, max_recs, max_workers=4, 
                            batch_size=8, n_test_views=128, iteration=0, printer_config_path=None,
                            enable_realtime=False):
    """
    Enhanced weakness analysis with real-time capture integration and adaptive sampling.
    """
    print(f"--- Real-time Weakness Analysis for run: {run_path} ---")
    print(f"--- Mode: '{mode.upper()}' | Iteration: {iteration} | Workers: {max_workers} ---")
    
    analysis_dir = os.path.join(run_path, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    snapshot_path = os.path.abspath(os.path.join(run_path, "scout_model/scout.ingp"))
    scene_json_path = os.path.abspath(os.path.join(run_path, "transforms.json"))
    official_script_path = os.path.abspath("./vendor/instant-ngp/scripts/run.py")

    if not os.path.exists(snapshot_path):
        print("❌ ERROR: Snapshot file not found.")
        exit()

    # Load scene configuration
    with open(scene_json_path, 'r') as f:
        training_data = json.load(f)
    
    positions = [np.array(frame["transform_matrix"])[:3, 3] for frame in training_data["frames"]]
    radius = np.mean([np.linalg.norm(p) for p in positions]) * 1.5
    camera_angle_x = training_data.get("camera_angle_x", 0.691111)
    
    print(f"Using camera sphere radius: {radius:.2f}")
    print(f"Analyzing {n_test_views} viewpoints with adaptive sampling...")

    # Initialize real-time capture system if available and enabled
    capture_system = None
    if REALTIME_AVAILABLE and enable_realtime and printer_config_path:
        try:
            with open(printer_config_path, 'r') as f:
                printer_config = json.load(f)
            capture_system = RealTimeCaptureSystem(printer_config)
            capture_system.iteration_count = iteration
            print("✅ Real-time capture system initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize real-time capture: {e}")

    # Generate adaptive viewpoints (changes each iteration to avoid overfitting)
    viewpoints = generate_viewpoints_adaptive(n_test_views, radius, iteration, capture_system)
    
    # Initialize renderer and analyzer
    renderer = ViewpointRenderer(snapshot_path, scene_json_path, official_script_path, 
                                camera_angle_x, analysis_dir)
    analyzer = WeaknessAnalyzer(mode)
    
    # Render viewpoints in parallel
    print("Step 2.1: Rendering adaptive viewpoints in parallel...")
    render_results = render_viewpoints_parallel(renderer, viewpoints, max_workers, batch_size)
    
    # Analyze rendered images
    print("Step 2.2: Analyzing rendered images...")
    view_analyses = []
    metric_scores = []
    view_positions_spherical = []
    
    for view_id, image_path, spherical_pos in tqdm(render_results, desc="Analyzing Images"):
        if image_path and os.path.exists(image_path):
            analysis_result = analyzer.analyze_image(image_path)
            if analysis_result:
                view_analyses.append({
                    'view_id': view_id,
                    'spherical_pos': spherical_pos,
                    'analysis': analysis_result,
                    'image_path': image_path
                })
                metric_scores.append(analysis_result['score'])
                view_positions_spherical.append(spherical_pos)
    
    print(f"Successfully analyzed {len(view_analyses)} viewpoints.")
    
    # Identify weak viewpoints with sophisticated ranking
    if mode == 'geometry':
        # For geometry, higher score = weaker
        threshold = np.percentile([a['analysis']['score'] for a in view_analyses], 75)
        weak_views = [v for v in view_analyses if v['analysis']['score'] > threshold]
    else:  # detail mode
        # For detail, lower score = weaker (blurrier)
        threshold = np.percentile([a['analysis']['score'] for a in view_analyses], 25)
        weak_views = [v for v in view_analyses if v['analysis']['score'] < threshold]
    
    print(f"Identified {len(weak_views)} weak viewpoints above/below threshold.")
    
    # Enhanced ranking with multiple factors
    def calculate_priority_score(view_analysis):
        """Calculate priority score considering multiple factors."""
        base_score = view_analysis['analysis']['score']
        
        # Distance from training cameras (prefer areas far from existing views)
        cam_pos = np.array([
            radius * np.cos(view_analysis['spherical_pos'][0]) * np.sin(view_analysis['spherical_pos'][1]),
            radius * np.sin(view_analysis['spherical_pos'][0]) * np.sin(view_analysis['spherical_pos'][1]),
            radius * np.cos(view_analysis['spherical_pos'][1])
        ])
        
        min_distance = float('inf')
        for pos in positions:
            distance = np.linalg.norm(cam_pos - pos)
            min_distance = min(min_distance, distance)
        
        distance_bonus = min_distance / radius  # Normalize by radius
        
        # Combine base weakness score with distance bonus
        priority = base_score + 0.3 * distance_bonus
        return priority
    
    # Rank weak views by priority
    for view in weak_views:
        view['priority_score'] = calculate_priority_score(view)
    
    weak_views.sort(key=lambda x: x['priority_score'], reverse=(mode == 'geometry'))
    
    # Generate enhanced heatmap
    if view_positions_spherical:
        print("Step 3: Generating enhanced uncertainty heatmap...")
        thetas, phis = zip(*view_positions_spherical)
        points = np.array([thetas, phis]).T
        grid_theta, grid_phi = np.mgrid[-np.pi:np.pi:500j, 0:np.pi:250j]
        grid_uncertainty = griddata(points, metric_scores, (grid_theta, grid_phi), method='cubic', fill_value=np.nan)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
        
        # Main heatmap
        cax1 = ax1.imshow(grid_uncertainty.T, extent=[360, 0, 180, 0], cmap='plasma', origin='upper', aspect='auto')
        ax1.set_title(f"Model Uncertainty Heatmap (Mode: {mode.upper()}, Iteration: {iteration})")
        ax1.set_xlabel("Theta (Azimuth)")
        ax1.set_ylabel("Phi (Inclination)")
        fig.colorbar(cax1, ax=ax1, label="Uncertainty Score")
        
        # Weak points overlay
        if weak_views:
            weak_thetas = [np.degrees(v['spherical_pos'][0]) % 360 for v in weak_views[:20]]  # Top 20
            weak_phis = [np.degrees(v['spherical_pos'][1]) for v in weak_views[:20]]
            ax1.scatter(weak_thetas, weak_phis, c='red', s=50, alpha=0.8, marker='x', linewidth=2)
        
        # Component analysis for geometry mode
        if mode == 'geometry' and weak_views:
            components = list(weak_views[0]['analysis']['components'].keys())
            component_scores = {comp: [] for comp in components}
            
            for view in view_analyses:
                for comp in components:
                    component_scores[comp].append(view['analysis']['components'][comp])
            
            # Average component scores
            avg_scores = [np.mean(component_scores[comp]) for comp in components]
            
            ax2.bar(range(len(components)), avg_scores)
            ax2.set_xticks(range(len(components)))
            ax2.set_xticklabels([c.replace('_', ' ').title() for c in components], rotation=45)
            ax2.set_title("Average Weakness Components")
            ax2.set_ylabel("Weakness Score")
        
        heatmap_path = os.path.join(analysis_dir, f"uncertainty_heatmap_{mode}_iter{iteration}.png")
        plt.tight_layout()
        plt.savefig(heatmap_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Enhanced heatmap saved to {heatmap_path}")
    
    # Find best candidate images (traditional dataset augmentation)
    if not weak_views:
        print("No weak viewpoints identified. Creating empty recommendations.")
        output_rec_path = os.path.join(run_path, "recommended_images.txt")
        open(output_rec_path, 'w').close()
        
        # Still save metadata for real-time capture
        if REALTIME_AVAILABLE and capture_system and enable_realtime:
            camera_positions = capture_system.extract_camera_positions_from_weakness_analysis([])
            positions_output = Path(run_path) / "camera_positions.json"
            capture_system.save_camera_positions_for_printer(camera_positions, str(positions_output))
        
        return
    
    print("Step 4: Processing recommendations for both traditional and real-time capture...")
    
    # Traditional dataset augmentation
    with open(full_dataset_json_path, 'r') as f:
        full_frames = json.load(f)['frames']
    source_base_dir = Path(full_dataset_json_path).parent

    # Get current training set
    with open(scene_json_path, 'r') as f:
        current_frames_data = json.load(f)['frames']
    current_absolute_paths = {frame['file_path'] for frame in current_frames_data}

    # Create candidate pool
    candidate_frames = []
    for frame in full_frames:
        abs_path = str((source_base_dir / Path(frame['file_path'])).resolve())
        if abs_path not in current_absolute_paths:
            candidate_frames.append(frame)

    print(f"Searching among {len(candidate_frames)} available candidate images...")

    # Match weak viewpoints to best candidates
    recommendations = []
    used_candidates = set()
    
    for weak_view in weak_views:
        weak_pos = np.array([
            radius * np.cos(weak_view['spherical_pos'][0]) * np.sin(weak_view['spherical_pos'][1]),
            radius * np.sin(weak_view['spherical_pos'][0]) * np.sin(weak_view['spherical_pos'][1]),
            radius * np.cos(weak_view['spherical_pos'][1])
        ])
        
        min_dist = float('inf')
        best_candidate = None
        
        for frame in candidate_frames:
            candidate_path = str((source_base_dir / Path(frame['file_path'])).resolve())
            if candidate_path in used_candidates:
                continue
                
            cam_pos = np.array(frame['transform_matrix'])[:3, 3]
            dist = np.linalg.norm(weak_pos - cam_pos)
            
            if dist < min_dist:
                min_dist = dist
                best_candidate = (candidate_path, frame)
        
        if best_candidate:
            recommendations.append({
                'path': best_candidate[0],
                'priority_score': weak_view['priority_score'],
                'weakness_components': weak_view['analysis']['components'],
                'distance': min_dist,
                'spherical_pos': weak_view['spherical_pos']  # Include for real-time capture
            })
            used_candidates.add(best_candidate[0])
    
    # Sort by priority and limit
    recommendations.sort(key=lambda x: x['priority_score'], reverse=(mode == 'geometry'))
    
    if max_recs > 0:
        recommendations = recommendations[:max_recs]
    
    # Save traditional recommendations
    output_rec_path = os.path.join(run_path, "recommended_images.txt")
    metadata_path = os.path.join(analysis_dir, "recommendation_metadata.json")
    
    with open(output_rec_path, 'w') as f:
        for rec in recommendations:
            f.write(rec['path'] + "\n")
    
    # Save detailed metadata
    metadata = {
        'mode': mode,
        'iteration': iteration,
        'n_test_views': n_test_views,
        'n_weak_views': len(weak_views),
        'n_recommendations': len(recommendations),
        'recommendations': recommendations,
        'adaptive_sampling': True,
        'realtime_enabled': enable_realtime and REALTIME_AVAILABLE
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Real-time capture integration
    if REALTIME_AVAILABLE and capture_system and enable_realtime:
        print("Step 5: Generating real-time capture positions...")
        
        # Convert weak viewpoints to camera positions
        camera_positions = capture_system.extract_camera_positions_from_weakness_analysis(weak_views[:max_recs])
        
        # Save camera positions for printer
        positions_output = Path(run_path) / "camera_positions.json"
        capture_system.save_camera_positions_for_printer(camera_positions, str(positions_output))
        
        # Optionally execute capture sequence
        if printer_config_path:
            with open(printer_config_path, 'r') as f:
                printer_config = json.load(f)
            
            if printer_config.get('auto_capture', False):
                print("Step 6: Executing automated capture sequence...")
                success = capture_system.execute_capture_sequence(camera_positions)
                if success:
                    print("✅ Automated capture completed successfully")
                else:
                    print("❌ Automated capture failed")
        
        # Increment iteration counter for next run
        capture_system.increment_iteration()
    
    print(f"Saved {len(recommendations)} recommendations to {output_rec_path}")
    print(f"Saved detailed metadata to {metadata_path}")
    
    # Print summary
    if recommendations:
        print("\nTop 5 Recommendations Summary:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"{i}. Priority Score: {rec['priority_score']:.3f}, Distance: {rec['distance']:.2f}")
    
    if REALTIME_AVAILABLE and enable_realtime:
        print("--- Enhanced real-time weakness analysis complete ---")
    else:
        print("--- Traditional weakness analysis complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced weakness analysis with real-time capture integration.")
    parser.add_argument("--run_path", required=True)
    parser.add_argument("--mode", choices=['geometry', 'detail'], default='geometry')
    parser.add_argument("--max_recommendations", type=int, default=10, help="Maximum number of new images to recommend per iteration. Set to 0 for unlimited.")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel rendering workers.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of viewpoints to render simultaneously.")
    parser.add_argument("--n_test_views", type=int, default=128, help="Total number of viewpoints to test.")
    parser.add_argument("--iteration", type=int, default=0, help="Current iteration number for adaptive sampling.")
    
    # Real-time capture arguments
    parser.add_argument("--enable_realtime", action='store_true', help="Enable real-time capture integration.")
    parser.add_argument("--printer_config", type=str, help="Path to printer configuration JSON file.")
    
    args = parser.parse_args()
    
    dataset_name = Path(args.run_path).parts[1]
    full_json_path = f"data/{dataset_name}/transforms_train.json"
    
    analyze_weakness_realtime(args.run_path, full_json_path, args.mode, args.max_recommendations, 
                            args.max_workers, args.batch_size, args.n_test_views, args.iteration,
                            args.printer_config, args.enable_realtime)