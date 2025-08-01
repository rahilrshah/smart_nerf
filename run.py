# run.py - The Master Pipeline Controller

import os
import subprocess
import argparse
import shutil
from pathlib import Path

def run_command(command, cwd="."):
    """Helper function to run a command and check for errors."""
    print(f"\n> Running command: {' '.join(command)}")
    try:
        # Use shell=True on Windows for batch files and commands
        is_shell_cmd = os.name == 'nt'
        subprocess.run(command, check=True, cwd=cwd, shell=is_shell_cmd)
    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR ---")
        print(f"Command failed with exit code {e.returncode}")
        print("Please check the error messages above.")
        print("--- Pipeline Halted ---")
        exit()
    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"Command not found: '{command[0]}'.")
        print("Is the program installed and in your PATH? Or is the path to the script correct?")
        print("--- Pipeline Halted ---")
        exit()

def pre_flight_checks(mode):
    """Check if necessary compiled files and directories exist."""
    print("--- Running Pre-flight Checks ---")
    checks_passed = True
    
    # Check for the official training script
    if mode in ['full', 'optimize_only']:
        if not os.path.exists("./vendor/instant-ngp/scripts/run.py"):
            print("❌ ERROR: Official training script not found at 'vendor/instant-ngp/scripts/run.py'.")
            print("  Please ensure the Instant-NGP submodule is correctly cloned and up-to-date.")
            checks_passed = False

    # Check for Neuralangelo only if we are planning to use it.
    if mode in ['full', 'mesh_only']:
        if not os.path.exists("./vendor/neuralangelo/projects/neuralangelo/train.py"):
            print("❌ ERROR: Neuralangelo training script not found.")
            print("  Please ensure the Neuralangelo submodule is correctly cloned.")
            checks_passed = False
        
    if checks_passed:
        print("✅ All checks passed.")
    else:
        exit()

def main(args):
    """Main function to orchestrate the pipeline."""
    
    pre_flight_checks(args.mode)
    
    dataset_source_path = f"data/{args.dataset}"
    if not os.path.exists(dataset_source_path):
        print(f"❌ ERROR: Dataset source directory not found at '{dataset_source_path}'.")
        exit()

    results_dir = f"results/{args.dataset}"
    os.makedirs(results_dir, exist_ok=True)
    final_optimized_json = os.path.join(results_dir, "optimized_transforms.json")

    if args.mode in ['full', 'optimize_only']:
        print("\n" + "="*50)
        print("    STARTING: Intelligent Dataset Optimization")
        print("="*50)

        # --- Step 1: Create Initial Sparse Subset (Our script) ---
        print("\n--- STEP 1: Creating Initial Sparse Dataset ---")
        current_run_name = "run_01_geom"
        current_run_path = f"experiments/{args.dataset}/{current_run_name}"
        
        cmd_step1 = [
            "python", "src/create_subset.py",
            "--dataset", args.dataset,
            "--num_images", str(args.initial_images),
            "--run_name", current_run_name
        ]
        run_command(cmd_step1)
        
        # --- Iterative Analysis and Augmentation Loop ---
        for i in range(1, args.max_iterations + 1):
            print(f"\n--- ITERATION {i}/{args.max_iterations}: Training & Analysis ---")
            
            # --- Step 2a: Train scout model (Official script) ---
            print("Step 2a: Training scout model with official Instant-NGP script...")
            scene_json_path = os.path.abspath(os.path.join(current_run_path, "transforms.json"))
            snapshot_path = os.path.abspath(os.path.join(current_run_path, "scout_model/scout.ingp"))
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            
            cmd_train = [
                "python",
                os.path.abspath("./vendor/instant-ngp/scripts/run.py"),
                "--scene", scene_json_path,
                "--save_snapshot", snapshot_path,
                "--n_steps", str(args.steps_geometry) # Use the official script's argument
            ]
            run_command(cmd_train)

            # --- Step 2b: Analyze model for weaknesses (Our script) ---
            print("Step 2b: Analyzing scout model for weaknesses...")
            analysis_mode = "geometry" if i < args.detail_start_iteration else "detail"
            cmd_analyze = [
                "python", "src/analyze_weakness.py",
                "--run_path", current_run_path,
                "--mode", analysis_mode
            ]
            run_command(cmd_analyze)

            # Step 3: Augment dataset (Our script)
            recommendations_file = os.path.join(current_run_path, "recommended_images.txt")
            if not os.path.exists(recommendations_file) or os.path.getsize(recommendations_file) == 0:
                print("\n✅ Analysis complete. No more images recommended. Exiting optimization loop.")
                break
            
            if i == args.max_iterations:
                print("\nReached max iterations. Exiting optimization loop.")
                break

            next_mode = "detail" if (i + 1) >= args.detail_start_iteration else "geom"
            next_run_name = f"run_{i+1:02d}_{next_mode}"
            
            cmd_step3 = [
                "python", "src/augment_dataset.py",
                "--previous_run", current_run_path,
                "--new_run_name", next_run_name
            ]
            run_command(cmd_step3)
            
            current_run_path = f"experiments/{args.dataset}/{next_run_name}"
        
        print("\n--- OPTIMIZATION COMPLETE ---")
        print(f"The best dataset was generated in: '{current_run_path}'")
        
        source_json = os.path.join(current_run_path, "transforms.json")
        shutil.copy(source_json, final_optimized_json)
        print(f"✅ Copied final optimized dataset to '{final_optimized_json}'")

    # The meshing logic (Step 4) can remain the same
    if args.mode in ['full', 'mesh_only']:
        print("\n" + "="*50)
        print("    STARTING: 3D Mesh Generation with Neuralangelo")
        print("="*50)
        
        if not os.path.exists(final_optimized_json):
            print(f"❌ ERROR: Cannot run meshing. Optimized file not found at '{final_optimized_json}'.")
            print("  Please run the 'full' or 'optimize_only' pipeline first.")
            exit()
            
        print("\n--- STEP 4: Launching Neuralangelo Pipeline ---")
        cmd_step4 = [
            "python", "src/run_neuralangelo.py",
            "--optimized_json", final_optimized_json,
            "--output_base", "results"
        ]
        run_command(cmd_step4)

    print("\n" + "="*50)
    print("      🎉 Pipeline Complete! 🎉")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master pipeline controller for the Intelligent 3D Capture project.")
    parser.add_argument("--dataset", required=True, help="Name of the dataset folder inside 'data/'.")
    parser.add_argument("--mode", choices=['full', 'optimize_only', 'mesh_only'], default='full', 
                        help="'full': Run optimization and meshing. 'optimize_only': Run only the dataset optimization. 'mesh_only': Run only meshing.")
    
    parser.add_argument("--initial_images", type=int, default=15, help="Number of images to start with.")
    parser.add_argument("--max_iterations", type=int, default=2, help="Maximum number of improvement iterations.")
    parser.add_argument("--detail_start_iteration", type=int, default=2, 
                        help="At which iteration to switch from 'geometry' to 'detail' analysis.")
    # Arguments for the official training script
    parser.add_argument("--steps_geometry", type=int, default=3500, help="Training steps for the geometry analysis phase.")
    
    args = parser.parse_args()
    main(args)