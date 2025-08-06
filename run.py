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
        # We only call Python scripts, so shell=False is safer.
        subprocess.run(command, check=True, cwd=cwd)
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

def pre_flight_checks():
    """Check if the official script we rely on exists."""
    print("--- Running Pre-flight Checks ---")
    if not os.path.exists("vendor/instant-ngp/scripts/run.py"):
        print("❌ ERROR: Official training script not found at 'vendor/instant-ngp/scripts/run.py'.")
        print("  Please ensure the Instant-NGP submodule is correctly cloned and all its files are present.")
        exit()
    print("✅ All checks passed.")

def main(args):
    """Main function to orchestrate the pipeline."""
    pre_flight_checks()
    dataset_source_path = f"data/{args.dataset}"
    if not os.path.exists(dataset_source_path):
        print(f"❌ ERROR: Dataset source directory not found at '{dataset_source_path}'."); exit()

    results_dir = f"results/{args.dataset}"; os.makedirs(results_dir, exist_ok=True)
    final_optimized_json = os.path.join(results_dir, "optimized_transforms.json")
    
    if args.mode in ['full', 'optimize_only']:
        print("\n" + "="*50); print("    STARTING: Intelligent Dataset Optimization"); print("="*50)
        
        print("\n--- STEP 1: Creating Initial Sparse Dataset ---")
        current_run_name = "run_01_geom"
        current_run_path = f"experiments/{args.dataset}/{current_run_name}"
        run_command(["python", "src/create_subset.py", "--dataset", args.dataset, "--num_images", str(args.initial_images), "--run_name", current_run_name])

        # We will loop one extra time to allow for the final augmentation
        for i in range(1, args.max_iterations + 2):
            # --- THE FINAL, CORRECTED LOOP LOGIC ---
            # If we are past the max number of *analysis* iterations, break.
            if i > args.max_iterations:
                print("\nReached max iterations. Finalizing.")
                break

            print(f"\n--- ITERATION {i}/{args.max_iterations}: Training & Analysis ---")
            
            # ... (Step 2a: Training is correct) ...
            scene_json_path = os.path.abspath(os.path.join(current_run_path, "transforms.json"))
            snapshot_path = os.path.abspath(os.path.join(current_run_path, "scout_model/scout.ingp"))
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            print("Step 2a: Training scout model with official Instant-NGP script...")
            cmd_train = ["python", os.path.abspath("./vendor/instant-ngp/scripts/run.py"), "--scene", scene_json_path, "--save_snapshot", snapshot_path, "--n_steps", "3500"]
            run_command(cmd_train)

            # ... (Step 2b: Analysis is correct) ...
            print("Step 2b: Analyzing scout model for weaknesses...")
            analysis_mode = "geometry" if i < args.detail_start_iteration else "detail"
            cmd_analyze = ["python", "src/analyze_weakness.py", "--run_path", current_run_path, "--mode", analysis_mode]
            run_command(cmd_analyze)

            recommendations_file = os.path.join(current_run_path, "recommended_images.txt")
            if not os.path.exists(recommendations_file) or os.path.getsize(recommendations_file) == 0:
                print("\n✅ Analysis complete. No more images recommended. Exiting optimization loop.")
                break

            # The loop will now naturally continue to augmentation after the final analysis.
            next_mode = "detail" if (i + 1) >= args.detail_start_iteration else "geom"
            next_run_name = f"run_{i+2:02d}_{next_mode}" # Use i+2 for naming the next folder
            run_command(["python", "src/augment_dataset.py", "--previous_run", current_run_path, "--new_run_name", next_run_name])
            current_run_path = f"experiments/{args.dataset}/{next_run_name}"
        
        print("\n--- OPTIMIZATION COMPLETE ---")
        # To get the final dataset, we must go back one step in the run path name
        final_run_index = int(current_run_path.split('_')[1]) -1
        final_mode = current_run_path.split('_')[2]
        final_run_path = os.path.join("experiments", args.dataset, f"run_{final_run_index:02d}_{final_mode}")
        
        print(f"The best dataset was generated in: '{final_run_path}'")
        source_json = os.path.join(final_run_path, "transforms.json")
        final_optimized_json = os.path.join("results", args.dataset, "optimized_transforms.json")
        os.makedirs(os.path.dirname(final_optimized_json), exist_ok=True)
        shutil.copy(source_json, final_optimized_json)
        print(f"✅ Copied final optimized dataset to '{final_optimized_json}'")

    # --- Part 2: Final Meshing with Neuralangelo (Fully Restored) ---
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
    parser = argparse.ArgumentParser(description="Master pipeline controller.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=['full', 'optimize_only', 'mesh_only'], default='full')
    parser.add_argument("--initial_images", type=int, default=15)
    parser.add_argument("--max_iterations", type=int, default=2)
    parser.add_argument("--detail_start_iteration", type=int, default=2)
    args = parser.parse_args()
    main(args)