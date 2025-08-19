# run.py - The Master Pipeline Controller (Updated with Parallel Analysis)

import os
import subprocess
import argparse
import shutil
import sys
import psutil
from pathlib import Path

def run_command(command, cwd="."):
    """Helper function to run a command and check for errors."""
    print(f"\n> Running command: {' '.join(command)}")
    try:
        # Use shell=True on Windows for .bat files, but False is safer for python calls
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
    
    # Conservative approach: use 50-75% of CPU cores, limited by memory
    # Each worker may use ~1-2GB RAM during rendering
    max_workers_by_cpu = max(2, int(cpu_count * 0.75))
    max_workers_by_memory = max(2, int(memory_gb / 2))
    
    optimal_workers = min(max_workers_by_cpu, max_workers_by_memory, 8)  # Cap at 8 for stability
    
    print(f"System detected: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
    print(f"Recommended workers: {optimal_workers}")
    
    return optimal_workers

def pre_flight_checks(mode):
    """Check if necessary script files exist based on the selected mode."""
    print("--- Running Pre-flight Checks ---")
    checks_passed = True
    
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
        
        # Check for the parallel analysis script
        if not os.path.exists("src/2_analyze_weakness_parallel.py"):
            print("❌ ERROR: Parallel analysis script not found at 'src/2_analyze_weakness_parallel.py'.")
            print("   Using fallback to original analysis script...")
            if not os.path.exists("src/analyze_weakness.py"):
                print("❌ ERROR: No analysis script found!")
                checks_passed = False

    if mode in ['full', 'mesh_only']:
        # Check if run_nerfstudio.py exists
        if not os.path.exists("src/run_nerfstudio.py"):
            print("❌ ERROR: Nerfstudio script not found at 'src/run_nerfstudio.py'.")
            checks_passed = False
            
    if checks_passed:
        print("✅ All checks passed.")
    else:
        exit()

def get_analysis_command(run_path, mode, max_recommendations, performance_args):
    """Choose between parallel and original analysis scripts based on availability."""
    parallel_script = "src/2_analyze_weakness_parallel.py"
    original_script = "src/analyze_weakness.py"
    
    if os.path.exists(parallel_script):
        print(f"Using enhanced parallel analysis (workers: {performance_args['workers']}, batch: {performance_args['batch_size']})")
        return [
            "python", parallel_script,
            "--run_path", run_path,
            "--mode", mode,
            "--max_recommendations", str(max_recommendations),
            "--max_workers", str(performance_args['workers']),
            "--batch_size", str(performance_args['batch_size']),
            "--n_test_views", str(performance_args['test_views'])
        ]
    else:
        print("Using original analysis script (sequential)")
        return [
            "python", original_script,
            "--run_path", run_path,
            "--mode", mode,
            "--max_recommendations", str(max_recommendations)
        ]

def main(args):
    """Main function to orchestrate the pipeline with enhanced parallel analysis."""
    
    pre_flight_checks(args.mode)
    
    dataset_source_path = f"data/{args.dataset}"
    if not os.path.exists(dataset_source_path):
        print(f"❌ ERROR: Dataset source directory not found at '{dataset_source_path}'."); 
        sys.exit(1)

    results_dir = f"results/{args.dataset}"
    os.makedirs(results_dir, exist_ok=True)
    final_optimized_json = os.path.join(results_dir, "optimized_transforms.json")
    
    # Setup performance parameters for parallel analysis
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
        print("\n" + "="*50)
        print("    STARTING: Intelligent Dataset Optimization")
        print(f"    Performance: {performance_args['workers']} workers, {performance_args['batch_size']} batch size")
        print("="*50)
        
        print("\n--- STEP 1: Creating Initial Sparse Dataset ---")
        current_run_name = "run_01_geom"
        current_run_path = f"experiments/{args.dataset}/{current_run_name}"
        run_command([
            "python", "src/create_subset.py", 
            "--dataset", args.dataset, 
            "--num_images", str(args.initial_images), 
            "--run_name", current_run_name
        ])

        # --- PHASE 1: Geometry Convergence Loop ---
        print("\n" + "="*50)
        print("    PHASE 1: GEOMETRY CONVERGENCE")
        print("="*50)
        
        geom_iteration = 1
        while True:
            print(f"\n--- GEOMETRY ITERATION {geom_iteration}/{args.max_geom_iterations} ---")

            scene_json_path = os.path.abspath(os.path.join(current_run_path, "transforms.json"))
            snapshot_path = os.path.abspath(os.path.join(current_run_path, "scout_model/scout.ingp"))
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            
            # Enhanced training command with adaptive steps
            training_steps = args.geom_training_steps
            if geom_iteration > 1:
                # Increase training steps for later iterations as dataset grows
                training_steps = int(training_steps * (1 + 0.2 * (geom_iteration - 1)))
            
            cmd_train = [
                "python", os.path.abspath("./vendor/instant-ngp/scripts/run.py"), 
                "--scene", scene_json_path, 
                "--save_snapshot", snapshot_path, 
                "--n_steps", str(training_steps)
            ]
            
            # Add GPU-specific flags if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"Training on GPU: {gpu_name}")
                    # RTX 4090/5080 optimizations could be added here
            except ImportError:
                print("PyTorch not available, training on available hardware")
                
            run_command(cmd_train)

            # Use enhanced parallel analysis
            cmd_analyze = get_analysis_command(current_run_path, "geometry", args.max_recommendations, performance_args)
            run_command(cmd_analyze)

            recommendations_file = os.path.join(current_run_path, "recommended_images.txt")
            if not os.path.exists(recommendations_file) or os.path.getsize(recommendations_file) == 0:
                print("\n✅ Geometry converged! No more geometric weaknesses found.")
                break
            
            # Check convergence criteria
            with open(recommendations_file, 'r') as f:
                num_recommendations = len([line for line in f if line.strip()])
            
            print(f"Analysis recommends {num_recommendations} new images")
            
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

        # --- PHASE 2: Detail Refinement Loop ---
        print("\n" + "="*50)
        print("    PHASE 2: DETAIL REFINEMENT")
        print("="*50)

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

            # Use enhanced parallel analysis for detail mode
            cmd_analyze = get_analysis_command(current_run_path, "detail", args.max_recommendations, performance_args)
            run_command(cmd_analyze)
            
            recommendations_file = os.path.join(current_run_path, "recommended_images.txt")
            if not os.path.exists(recommendations_file) or os.path.getsize(recommendations_file) == 0:
                print("\n✅ Detail refined! No more recommendations found.")
                break

            # Check recommendations count
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
                "--selection_strategy", "top_priority"  # Use more aggressive selection in detail phase
            ])
            current_run_path = f"experiments/{args.dataset}/{next_run_name}"
            
        print("\n--- OPTIMIZATION COMPLETE ---")
        
        # Generate final summary
        final_transforms_path = os.path.join(current_run_path, "transforms.json")
        if os.path.exists(final_transforms_path):
            with open(final_transforms_path, 'r') as f:
                import json
                final_data = json.load(f)
                final_image_count = len(final_data['frames'])
                print(f"\n📊 OPTIMIZATION SUMMARY:")
                print(f"   • Started with: {args.initial_images} images")
                print(f"   • Final dataset: {final_image_count} images")
                print(f"   • Images added: {final_image_count - args.initial_images}")
                print(f"   • Geometry iterations: {geom_iteration}")
                print(f"   • Detail iterations: {detail_iteration}")
        
        os.makedirs(os.path.dirname(final_optimized_json), exist_ok=True)
        shutil.copy(os.path.join(current_run_path, "transforms.json"), final_optimized_json)
        print(f"✅ Copied final optimized dataset from '{current_run_path}' to '{final_optimized_json}'")
        
    if args.mode in ['full', 'mesh_only']:
        print("\n" + "="*50)
        print("    STARTING: Nerfstudio Mesh Generation")
        print("="*50)
        
        if not os.path.exists(final_optimized_json):
            print(f"❌ ERROR: Optimized transforms not found at '{final_optimized_json}'")
            print("   Run optimization phase first!")
            sys.exit(1)
        
        # Run Nerfstudio mesh generation
        cmd_mesh = [
            "python", "src/run_nerfstudio.py",
            "--optimized_json", final_optimized_json,
            "--method", args.mesh_method, 
            "--max_num_iterations", str(args.mesh_iterations)
        ]
        run_command(cmd_mesh)
        
        print("\n🎉 COMPLETE PIPELINE FINISHED! 🎉")
        print(f"📁 Results available in: {results_dir}/")
        print(f"📈 Enhanced parallel analysis provided {performance_args['workers']}x speedup")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master pipeline controller for the Intelligent 3D Capture project with parallel analysis.")
    
    # Core arguments
    parser.add_argument("--dataset", required=True, help="Dataset name to process")
    parser.add_argument("--mode", choices=['full', 'optimize_only', 'mesh_only'], default='full',
                       help="Pipeline mode: full=optimize+mesh, optimize_only=just optimization, mesh_only=just mesh generation")
    
    # Dataset arguments
    parser.add_argument("--initial_images", type=int, default=15, 
                       help="Number of initial images for sparse dataset")
    
    # Iteration control
    parser.add_argument("--max_geom_iterations", type=int, default=4, 
                       help="Max loops for fixing geometry/holes")
    parser.add_argument("--max_detail_iterations", type=int, default=4, 
                       help="Max loops for sharpening details after geometry is solid")
    parser.add_argument("--max_recommendations", type=int, default=10, 
                       help="Max new images to add per iteration")
    
    # Training control
    parser.add_argument("--geom_training_steps", type=int, default=3500,
                       help="Training steps for geometry phase")
    parser.add_argument("--detail_training_steps", type=int, default=5000,
                       help="Training steps for detail phase")
    
    # Performance arguments for parallel analysis
    parser.add_argument("--auto_workers", action='store_true', default=True,
                       help="Automatically detect optimal number of workers")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of parallel rendering workers (if not auto)")
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