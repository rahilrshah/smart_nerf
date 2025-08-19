# src/run_nerfstudio.py - Simplified Nerfstudio runner

import argparse
import subprocess
import os
import sys
import json
from pathlib import Path
import shutil
from PIL import Image
import math

def run_command(command, step_name):
    """Run subprocess with basic error handling."""
    print(f"\n--- {step_name} ---")
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {step_name} completed successfully")
            return True
        else:
            print(f"❌ {step_name} failed")
            print("STDERR:", result.stderr[-500:] if result.stderr else "None")
            return False
    except Exception as e:
        print(f"❌ {step_name} failed with exception: {e}")
        return False

def setup_nerfstudio_data(optimized_json_path, dataset_name):
    """Prepare data for nerfstudio with proper image-frame matching."""
    json_path = Path(optimized_json_path)
    if not json_path.exists():
        print(f"ERROR: transforms file not found at '{json_path}'")
        sys.exit(1)
        
    nerfstudio_data_dir = json_path.parent / "nerfstudio_data"
    nerfstudio_data_dir.mkdir(exist_ok=True)
    
    with open(json_path, 'r') as f:
        transforms_data = json.load(f)
    
    # Find source images directory
    possible_dirs = [
        Path("data") / dataset_name / "images",
        Path("data") / dataset_name / "train", 
        Path("data") / dataset_name
    ]
    
    source_dir = None
    for img_dir in possible_dirs:
        if img_dir.exists():
            images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            if images:
                source_dir = img_dir
                break
    
    if not source_dir:
        print(f"ERROR: No images found in {possible_dirs}")
        sys.exit(1)
    
    # Create target images directory
    target_images_dir = nerfstudio_data_dir / "images"
    target_images_dir.mkdir(exist_ok=True)
    
    # FIXED: Only copy images that have corresponding frames
    source_images_map = {}
    for ext in ['*.jpg', '*.png', '*.JPG', '*.PNG']:
        for img_path in source_dir.glob(ext):
            # Try different stem variations to match
            stems_to_try = [
                img_path.stem,
                img_path.stem.replace('r_', ''),  # Handle r_44 -> 44
                f"r_{img_path.stem}",  # Handle 44 -> r_44
            ]
            for stem in stems_to_try:
                source_images_map[stem] = img_path
    
    print(f"Found {len(source_images_map)} source images with various naming patterns")
    
    # Get image dimensions from first available image
    sample_image = next(iter(source_images_map.values()))
    with Image.open(sample_image) as img:
        w, h = img.size
    
    # Calculate intrinsics
    camera_angle_x = transforms_data.get("camera_angle_x", 0.6911)
    fl_x = 0.5 * w / math.tan(0.5 * camera_angle_x)
    print(f"Image size: {w}x{h}, focal length: {fl_x:.2f}")
    
    # Process frames and copy only needed images
    valid_frames = []
    copied_images = set()
    
    for frame in transforms_data['frames']:
        original_path = frame['file_path']
        
        # Try to find matching image with multiple strategies
        stems_to_try = [
            Path(original_path).stem,
            Path(original_path).name,  # Full filename
            str(Path(original_path)).replace('/', '_').replace('\\', '_'),  # Path as stem
        ]
        
        # Also try absolute path matching
        if original_path.startswith('/') or ':\\' in original_path:
            stems_to_try.append(Path(original_path).name)
            stems_to_try.append(Path(original_path).stem)
        
        found_image = None
        found_stem = None
        
        for stem in stems_to_try:
            if stem in source_images_map:
                found_image = source_images_map[stem]
                found_stem = stem
                break
        
        if found_image and found_stem not in copied_images:
            # Copy the image to target directory
            target_image_name = f"{found_stem}{found_image.suffix}"
            target_path = target_images_dir / target_image_name
            
            try:
                shutil.copy2(found_image, target_path)
                copied_images.add(found_stem)
                
                # Update frame data
                frame['file_path'] = f"images/{target_image_name}"
                frame.update({
                    'fl_x': fl_x, 'fl_y': fl_x,
                    'cx': w/2, 'cy': h/2,
                    'w': w, 'h': h
                })
                valid_frames.append(frame)
                
            except Exception as e:
                print(f"⚠️ Failed to copy {found_image}: {e}")
        else:
            print(f"⚠️ No matching image found for frame: {Path(original_path).stem}")
    
    print(f"✅ Copied {len(copied_images)} images for {len(valid_frames)} valid frames")
    print(f"📊 Frame matching: {len(valid_frames)}/{len(transforms_data['frames'])} frames kept")
    
    # Update transforms data
    transforms_data['frames'] = valid_frames
    transforms_data['camera_model'] = 'OPENCV'
    
    # Add debug metadata
    transforms_data['_nerfstudio_setup_metadata'] = {
        'source_dir': str(source_dir),
        'original_frames': len(transforms_data.get('frames', [])),
        'valid_frames': len(valid_frames),
        'images_copied': len(copied_images)
    }
    
    # Save transforms
    transforms_file = nerfstudio_data_dir / "transforms.json"
    with open(transforms_file, 'w') as f:
        json.dump(transforms_data, f, indent=2)
    
    # Final verification
    actual_images = list(target_images_dir.glob("*.jpg")) + list(target_images_dir.glob("*.png"))
    print(f"🔍 Final verification:")
    print(f"   - Frames in transforms.json: {len(valid_frames)}")
    print(f"   - Images in directory: {len(actual_images)}")
    
    if len(valid_frames) == len(actual_images):
        print("   ✅ SUCCESS: Frame count matches image count!")
    else:
        print(f"   ⚠️ MISMATCH: {len(valid_frames)} frames vs {len(actual_images)} images")
    
    return str(nerfstudio_data_dir)

def main():
    parser = argparse.ArgumentParser(description="Simplified Nerfstudio runner")
    parser.add_argument('--optimized_json', required=True, help="Path to transforms.json")
    parser.add_argument('--method', default='nerfacto', help="Nerfstudio method")
    parser.add_argument('--max_num_iterations', type=int, default=20000, help="Training iterations")
    
    args = parser.parse_args()
    
    json_path = Path(args.optimized_json)
    dataset_name = json_path.parent.name
    
    print(f"Dataset: {dataset_name}")
    print(f"Method: {args.method}")
    print(f"Iterations: {args.max_num_iterations}")
    
    # Setup data
    data_dir = setup_nerfstudio_data(args.optimized_json, dataset_name)
    
    # Create output directory
    mesh_output_dir = Path(f"results/{dataset_name}/meshes")
    mesh_output_dir.mkdir(parents=True, exist_ok=True)
    
    experiment_name = f"{dataset_name}_{args.method}"
    
    # Train model
    train_command = [
        "ns-train", args.method,
        "--data", data_dir,
        "--experiment-name", experiment_name,
        "--max-num-iterations", str(args.max_num_iterations),
        "--vis", "tensorboard"
    ]
    
    if not run_command(train_command, "Training"):
        print("Training failed")
        sys.exit(1)
    
    # Find config
    configs_dir = Path("outputs") / experiment_name / args.method
    if not configs_dir.exists():
        print(f"Training output not found at {configs_dir}")
        sys.exit(1)
    
    # Get latest run
    run_dirs = sorted([d for d in configs_dir.iterdir() if d.is_dir()], 
                     key=lambda x: x.stat().st_mtime, reverse=True)
    if not run_dirs:
        print("No training runs found")
        sys.exit(1)
    
    config_path = run_dirs[0] / "config.yml"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    
    print(f"✅ Training complete: {config_path}")
    
    # Export mesh using simplified approach
    final_mesh_path = mesh_output_dir / f"{dataset_name}_mesh.ply"
    
    export_command = [
        "ns-export", "poisson",
        "--load-config", str(config_path),
        "--output-dir", str(mesh_output_dir)
    ]
    
    success = run_command(export_command, "Mesh Export")
    
    if success:
        # Find the exported mesh
        mesh_files = list(mesh_output_dir.glob("*.ply"))
        if mesh_files:
            # Rename to standard name if needed
            latest_mesh = max(mesh_files, key=lambda x: x.stat().st_mtime)
            if latest_mesh != final_mesh_path:
                shutil.move(str(latest_mesh), str(final_mesh_path))
            
            print(f"🎉 Mesh exported: {final_mesh_path}")
            print(f"File size: {final_mesh_path.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print("❌ No mesh file found after export")
    else:
        print("❌ Mesh export failed")
        print("\n💡 Try manual export:")
        print(f"   ns-export poisson --load-config {config_path} --output-dir {mesh_output_dir}")

if __name__ == '__main__':
    main()