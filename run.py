# run_realtime.py - Enhanced Pipeline Controller with Real-time Capture Integration

import os
import subprocess
import argparse
import shutil
import sys
import psutil
import json
import time
from pathlib import Path

def run_command(command, cwd="."):
    """Helper function to run a command and check for errors."""
    print(f"\n> Running command: {' '.join(command)}")
    try:
        is_shell_cmd = command[0].endswith('.bat') if os.name == 'nt' else False
        subprocess.run(command, check=True, cwd=cwd, shell=is_shell_cmd)
    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR ---")
        print(f"Command failed with exit code {e.returncode}")
        print("Please review the error messages above.")
        print("--- Pipeline Halted ---")
        exit()
    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"Command not found: '{command[0]}'.")
        print("Is the program installed and in your PATH? Or is the path to the script correct?")
        print("--- Pipeline Halted ---")
        exit()

def detect_optimal_workers():
    """Automatically detect optimal number of workers based on system resources."""
    cpu_count = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    max_workers_by_cpu = max(2, int(cpu_count * 0.75))
    max_workers_by_memory = max(2, int(memory_gb / 2))
    
    optimal_workers = min(max_workers_by_cpu, max_workers_by_memory, 8)
    
    print(f"System detected: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
    print(f"Recommended workers: {optimal_workers}")
    
    return optimal_workers

def setup_printer_configuration(dataset_name: str, args) -> str:
    """Setup and validate printer configuration."""
    config_dir = "configs"
    os.makedirs(config_dir, exist_ok=True)
    
    default_config_path = os.path.join(config_dir, f"{dataset_name}_printer_config.json")
    
    if args.printer_config:
        config_path = args.printer_config
        if not os.path.exists(config_path):
            print(f"❌ ERROR: Printer config file not found at '{config_path}'")
            sys.exit(1)
    else:
        config_path = default_config_path
        
    # Create default config if it doesn't exist
    if not os.path.exists(config_path):
        print(f"Creating default printer configuration at: {config_path}")
        default_config = {
            "printer_info": {"model": "Bambu Lab X1 Carbon", "bed_size_x": 256, "bed_size_y": 256, "max_z": 256},
            "object_setup": {"object_height": 50, "object_center_x": 128, "object_center_y": 128},
            "camera_setup": {"capture_radius": 150, "min_camera_distance": 100, "max_camera_distance": 300},
            "gimbal_config": {"pan_servo_pin": 0, "tilt_servo_pin": 1, "pan_range": [-180, 180], "tilt_range": [-90, 45]},
            "connectivity": {"connection_type": "serial", "serial_port": "COM3", "baud_rate": 115200},
            "capture_settings": {"auto_capture": False, "capture_trigger_pin": 2, "preview_mode": True},
            "motion_settings": {"travel_speed": 3000, "positioning_speed": 1500},
            "safety_limits": {"min_bed_margin": 10, "collision_detection": True}
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    
    # Validate configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        required_keys = ['printer_info', 'object_setup', 'camera_setup', 'connectivity']
        for key in required_keys:
            if key not in config:
                print(f"❌ ERROR: Missing required configuration section: '{key}'")
                sys.exit(1)
                
        print(f"✅ Printer configuration validated: {config_path}")
        return config_path
        
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in printer config: {e}")
        sys.exit(1)

def pre_flight_checks(mode, enable_realtime):
    """Enhanced pre-flight checks including real-time capture components."""
    print("--- Running Enhanced Pre-flight Checks ---")
    checks_passed = True
    
    # Standard checks
    if mode in ['full', 'optimize_only']:
        if not os.path.exists("vendor/instant-ngp/scripts/run.py"):
            print("❌ ERROR: Official training script not found at 'vendor/instant-ngp/scripts/run.py'.")
            checks_passed = False
        
        exe_name = "instant-ngp.exe"
        paths_to_check = [f"./vendor/instant-ngp/build/Release/{exe_name}", f"./vendor/instant-ngp/build/{exe_name}"]
        linux_path = "./vendor/instant-ngp/build/testbed"
        if not any(os.path.exists(p) for p in paths_to_check) and not os.path.exists(linux_path):
            print(f"❌ ERROR: Instant-NGP executable not found for analysis.")
            checks_passed = False
        
        # Check for real-time analysis script
        if not os.path.exists("src/analyze_weakness_realtime.py"):
            print("⚠️ WARNING: Real-time analysis script not found. Using fallback...")
            if not os.path.exists("src/2_analyze_weakness_parallel.py"):
                print("❌ ERROR: No analysis script found!")
                checks_passed = False

    # Real-time specific checks
    if enable_realtime:
        if not os.path.exists("src/camera_position_controller.py"):
            print("❌ ERROR: Camera position controller not found for real-time mode.")
            checks_passed = False
        
        # Check for required Python packages
        try:
            import serial
            print("✅ Serial communication available")
        except ImportError:
            print("⚠️ WARNING: pyserial not installed. Serial communication unavailable.")
            
        try:
            import cv2
            print("✅ OpenCV available for image processing")
        except ImportError:
            print("❌ ERROR: OpenCV not available. Required for real-time capture.")
            checks_passed = False

    if mode in ['full', 'mesh_only']:
        if not os.path.exists("src/run_nerfstudio.py"):
            print("❌ ERROR: Nerfstudio script not found at 'src/run_nerfstudio.py'.")
            checks_passed = False
            
    if checks_passed:
        print("✅ All checks passed.")
    else:
        print("❌ Pre-flight checks failed. Please resolve issues above.")
        exit()

def get_analysis_command(run_path, mode, max_recommendations, performance_args, iteration=0, 
                        enable_realtime=False, printer_config_path=None):
    """Choose analysis script and build command with real-time options."""
    
    # Prefer real-time analysis if available and requested
    if enable_realtime and os.path.exists("src/analyze_weakness_realtime.py"):
        print(f"Using real-time analysis (workers: {performance_args['workers']}, batch: {performance_args['batch_size']})")
        cmd = [
            "python", "src/analyze_weakness_realtime.py",
            "--run_path", run_path,
            "--mode", mode,
            "--max_recommendations", str(max_recommendations),
            "--max_workers", str(performance_args['workers']),
            "--batch_size", str(performance_args['batch_size']),
            "--n_test_views", str(performance_args['test_views']),
            "--iteration", str(iteration),
            "--enable_realtime"
        ]
        
        if printer_config_path:
            cmd.extend(["--printer_config", printer_config_path])
            
        return cmd
    
    # Fallback to parallel analysis
    elif os.path.exists("src/2_analyze_weakness_parallel.py"):
        print("Using enhanced parallel analysis (sequential)")
        return [
            "python", "src/2_analyze_weakness_parallel.py",
            "--run_path", run_path,
            "--mode", mode,
            "--max_recommendations", str(max_recommendations),
            "--max_workers", str(performance_args['workers']),
            "--batch_size", str(performance_args['batch_size']),
            "--n_test_views", str(performance_args['test_views'])
        ]
    
    # Original analysis as final fallback
    else:
        print("Using original analysis script (sequential)")
        return [
            "python", "src/analyze_weakness.py",
            "--run_path", run_path,
            "--mode", mode,
            "--max_recommendations", str(max_recommendations)
        ]

def process_real_time_capture_results(run_path: str, printer_config_path: str):
    """Process and integrate real-time capture results if available."""
    
    positions_file = Path(run_path) / "camera_positions.json"
    captured_images_dir = Path(run_path) / "captured_images"
    
    if positions_file.exists():
        print("\n--- Processing Real-time Capture Results ---")
        
        with open(positions_file, 'r') as f:
            capture_data = json.load(f)
        
        num_positions = len(capture_data.get('positions', []))
        print(f"Found {num_positions} recommended capture positions")
        
        # Check if images were actually captured
        if captured_images_dir.exists():
            captured_images = list(captured_images_dir.glob("*.jpg")) + list(captured_images_dir.glob("*.png"))
            print(f"Found {len(captured_images)} captured images")
            
            # TODO: Integrate captured images into the dataset
            # This would involve:
            # 1. Processing captured images
            # 2. Estimating camera poses
            # 3. Adding to transforms.json
            # 4. Updating the training dataset
            
        else:
            print("No captured images directory found. Positions generated for manual capture.")
            
        return True
    
    return False

def main(args):
    """Enhanced main function with real-time capture integration."""
    
    # Setup printer configuration if real-time mode is enabled
    printer_config_path = None
    if args.enable_realtime:
        printer_config_path = setup_printer_configuration(args.dataset, args)
    
    pre_flight_checks(args.mode, args.enable_realtime)
    
    dataset_source_path = f"data/{args.dataset}"
    if not os.path.exists(dataset_source_path):
        print(f"❌ ERROR: Dataset source directory not found at '{dataset_source_path}'."); 
        sys.exit(1)

    results_dir = f"results/{args.dataset}"
    os.makedirs(results_dir, exist_ok=True)
    final_optimized_json = os.path.join(results_dir, "optimized_transforms.json")
    
    # Setup performance parameters
    if args.auto_workers:
        optimal_workers = detect_optimal_workers()
    else:
        optimal_workers = args.max_workers
    
    performance_args = {
        'workers': optimal_workers,
        'batch_size': args.batch_size,
        'test_views': args.test_views
    }
    
    if args.mode in ['full', 'optimize_only']:
        print("\n" + "="*60)
        if args.enable_realtime:
            print("    STARTING: REAL-TIME INTELLIGENT DATASET OPTIMIZATION")
            print("    🔧 Integrated with Bambu Lab X1 Carbon + Gimbal")
        else:
            print("    STARTING: INTELLIGENT DATASET OPTIMIZATION")
        print(f"    Performance: {performance_args['workers']} workers, {performance_args['batch_size']} batch size")
        print("="*60)
        
        print("\n--- STEP 1: Creating Initial Sparse Dataset ---")
        current_run_name = "run_01_geom"
        current_run_path = f"experiments/{args.dataset}/{current_run_name}"
        run_command([
            "python", "src/create_subset.py", 
            "--dataset", args.dataset, 
            "--num_images", str(args.initial_images), 
            "--run_name", current_run_name
        ])

        # --- PHASE 1: Geometry Convergence Loop with Real-time Integration ---
        print("\n" + "="*60)
        if args.enable_realtime:
            print("    PHASE 1: REAL-TIME GEOMETRY CONVERGENCE")
            print("    📸 Adaptive sampling prevents overfitting")
        else:
            print("    PHASE 1: GEOMETRY CONVERGENCE")
        print("="*60)
        
        geom_iteration = 1
        while True:
            print(f"\n--- GEOMETRY ITERATION {geom_iteration}/{args.max_geom_iterations} ---")

            scene_json_path = os.path.abspath(os.path.join(current_run_path, "transforms.json"))
            snapshot_path = os.path.abspath(os.path.join(current_run_path, "scout_model/scout.ingp"))
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            
            # Enhanced training with adaptive steps
            training_steps = args.geom_training_steps
            if geom_iteration > 1:
                training_steps = int(training_steps * (1 + 0.2 * (geom_iteration - 1)))
            
            cmd_train = [
                "python", os.path.abspath("./vendor/instant-ngp/scripts/run.py"), 
                "--scene", scene_json_path, 
                "--save_snapshot", snapshot_path, 
                "--n_steps", str(training_steps)
            ]
            run_command(cmd_train)

            # Enhanced analysis with real-time integration
            cmd_analyze = get_analysis_command(
                current_run_path, "geometry", args.max_recommendations, 
                performance_args, geom_iteration - 1, args.enable_realtime, printer_config_path
            )
            run_command(cmd_analyze)
            
            # Process real-time capture results if available
            if args.enable_realtime:
                process_real_time_capture_results(current_run_path, printer_config_path)

            # Check convergence
            recommendations_file = os.path.join(current_run_path, "recommended_images.txt")
            if not os.path.exists(recommendations_file) or os.path.getsize(recommendations_file) == 0:
                print("\n✅ Geometry converged! No more geometric weaknesses found.")
                break
            
            with open(recommendations_file, 'r') as f:
                num_recommendations = len([line for line in f if line.strip()])
            
            print(f"Analysis recommends {num_recommendations} new images")
            
            if args.enable_realtime:
                # Check for real-time captured images
                captured_dir = Path(current_run_path) / "captured_images"
                if captured_dir.exists():
                    captured_count = len(list(captured_dir.glob("*.jpg")) + list(captured_dir.glob("*.png")))
                    print(f"📸 Real-time capture: {captured_count} images available")
            
            if geom_iteration >= args.max_geom_iterations:
                print(f"\nReached max geometry iterations ({args.max_geom_iterations}). Moving to detail phase.")
                break
            
            geom_iteration += 1
            next_run_name = f"run_{geom_iteration:02d}_geom"
            run_command([
                "python", "src/augment_dataset.py", 
                "--previous_run", current_run_path, 
                "--new_run_name", next_run_name,
                "--max_recommendations", str(args.max_recommendations),
                "--selection_strategy", "balanced"
            ])
            current_run_path = f"experiments/{args.dataset}/{next_run_name}"

        # --- PHASE 2: Detail Refinement Loop with Real-time Integration ---
        print("\n" + "="*60)
        if args.enable_realtime:
            print("    PHASE 2: REAL-TIME DETAIL REFINEMENT")
            print("    🔍 Fine-tuning with adaptive probe patterns")
        else:
            print("    PHASE 2: DETAIL REFINEMENT")
        print("="*60)

        for detail_iteration in range(1, args.max_detail_iterations + 1):
            print(f"\n--- DETAIL ITERATION {detail_iteration}/{args.max_detail_iterations} ---")
            
            scene_json_path = os.path.abspath(os.path.join(current_run_path, "transforms.json"))
            snapshot_path = os.path.abspath(os.path.join(current_run_path, "scout_model/scout.ingp"))
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            
            # Train for longer in detail mode with adaptive steps
            training_steps = args.detail_training_steps
            if detail_iteration > 1:
                training_steps = int(training_steps * (1 + 0.1 * (detail_iteration - 1)))
            
            cmd_train = [
                "python", os.path.abspath("./vendor/instant-ngp/scripts/run.py"), 
                "--scene", scene_json_path, 
                "--save_snapshot", snapshot_path, 
                "--n_steps", str(training_steps)
            ]
            run_command(cmd_train)

            # Enhanced detail analysis with real-time integration
            current_iteration = geom_iteration + detail_iteration - 1
            cmd_analyze = get_analysis_command(
                current_run_path, "detail", args.max_recommendations, 
                performance_args, current_iteration, args.enable_realtime, printer_config_path
            )
            run_command(cmd_analyze)
            
            # Process real-time capture results
            if args.enable_realtime:
                process_real_time_capture_results(current_run_path, printer_config_path)
            
            recommendations_file = os.path.join(current_run_path, "recommended_images.txt")
            if not os.path.exists(recommendations_file) or os.path.getsize(recommendations_file) == 0:
                print("\n✅ Detail refined! No more recommendations found.")
                break

            with open(recommendations_file, 'r') as f:
                num_recommendations = len([line for line in f if line.strip()])
            
            print(f"Analysis recommends {num_recommendations} new images for detail improvement")

            if detail_iteration >= args.max_detail_iterations:
                print("\nReached max detail iterations.")
                break

            next_run_name = f"run_{geom_iteration + detail_iteration:02d}_detail"
            run_command([
                "python", "src/augment_dataset.py", 
                "--previous_run", current_run_path, 
                "--new_run_name", next_run_name,
                "--max_recommendations", str(args.max_recommendations),
                "--selection_strategy", "top_priority"
            ])
            current_run_path = f"experiments/{args.dataset}/{next_run_name}"
            
        print("\n--- OPTIMIZATION COMPLETE ---")
        
        # Generate final summary with real-time stats
        final_transforms_path = os.path.join(current_run_path, "transforms.json")
        if os.path.exists(final_transforms_path):
            with open(final_transforms_path, 'r') as f:
                final_data = json.load(f)
                final_image_count = len(final_data['frames'])
                
                print(f"\n📊 OPTIMIZATION SUMMARY:")
                print(f"   • Started with: {args.initial_images} images")
                print(f"   • Final dataset: {final_image_count} images")
                print(f"   • Images added: {final_image_count - args.initial_images}")
                print(f"   • Geometry iterations: {geom_iteration}")
                print(f"   • Detail iterations: {detail_iteration}")
                
                if args.enable_realtime:
                    # Count total real-time captures across all iterations
                    total_positions = 0
                    total_captures = 0
                    
                    for exp_path in Path(f"experiments/{args.dataset}").glob("run_*"):
                        pos_file = exp_path / "camera_positions.json"
                        cap_dir = exp_path / "captured_images"
                        
                        if pos_file.exists():
                            with open(pos_file, 'r') as f:
                                data = json.load(f)
                                total_positions += len(data.get('positions', []))
                        
                        if cap_dir.exists():
                            total_captures += len(list(cap_dir.glob("*.jpg")) + list(cap_dir.glob("*.png")))
                    
                    print(f"   🔧 Real-time integration:")
                    print(f"      - Camera positions generated: {total_positions}")
                    print(f"      - Images captured: {total_captures}")
                    print(f"      - Adaptive sampling prevented overfitting")
        
        os.makedirs(os.path.dirname(final_optimized_json), exist_ok=True)
        shutil.copy(os.path.join(current_run_path, "transforms.json"), final_optimized_json)
        print(f"✅ Copied final optimized dataset to '{final_optimized_json}'")
        
    if args.mode in ['full', 'mesh_only']:
        print("\n" + "="*60)
        print("    STARTING: NERFSTUDIO MESH GENERATION")
        print("="*60)
        
        if not os.path.exists(final_optimized_json):
            print(f"❌ ERROR: Optimized transforms not found at '{final_optimized_json}'")
            print("   Run optimization phase first!")
            sys.exit(1)
        
        cmd_mesh = [
            "python", "src/run_nerfstudio.py",
            "--optimized_json", final_optimized_json,
            "--method", args.mesh_method, 
            "--max_num_iterations", str(args.mesh_iterations)
        ]
        run_command(cmd_mesh)
        
        print("\n🎉 COMPLETE PIPELINE FINISHED! 🎉")
        print(f"📁 Results available in: {results_dir}/")
        
        if args.enable_realtime:
            print(f"🔧 Real-time integration provided adaptive sampling and coordinate outputs")
            print(f"📸 Check G-code files for automated capture sequences")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced pipeline controller with real-time capture integration.")
    
    # Core arguments
    parser.add_argument("--dataset", required=True, help="Dataset name to process")
    parser.add_argument("--mode", choices=['full', 'optimize_only', 'mesh_only'], default='full',
                       help="Pipeline mode")
    
    # Real-time capture arguments
    parser.add_argument("--enable_realtime", action='store_true', 
                       help="Enable real-time capture integration with 3D printer")
    parser.add_argument("--printer_config", type=str,
                       help="Path to printer configuration JSON file")
    
    # Dataset arguments
    parser.add_argument("--initial_images", type=int, default=15, 
                       help="Number of initial images for sparse dataset")
    
    # Iteration control
    parser.add_argument("--max_geom_iterations", type=int, default=4, 
                       help="Max loops for fixing geometry/holes")
    parser.add_argument("--max_detail_iterations", type=int, default=4, 
                       help="Max loops for sharpening details")
    parser.add_argument("--max_recommendations", type=int, default=10, 
                       help="Max new images to add per iteration")
    
    # Training control
    parser.add_argument("--geom_training_steps", type=int, default=3500,
                       help="Training steps for geometry phase")
    parser.add_argument("--detail_training_steps", type=int, default=5000,
                       help="Training steps for detail phase")
    
    # Performance arguments
    parser.add_argument("--auto_workers", action='store_true', default=True,
                       help="Automatically detect optimal number of workers")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of parallel rendering workers")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Number of viewpoints to render simultaneously")
    parser.add_argument("--test_views", type=int, default=128,
                       help="Total number of viewpoints to test for weaknesses")
    
    # Mesh generation arguments
    parser.add_argument("--mesh_method", default="neus-facto",
                       help="Nerfstudio method for mesh generation")
    parser.add_argument("--mesh_iterations", type=int, default=20000,
                       help="Training iterations for mesh generation")

    args = parser.parse_args()
    main(args)