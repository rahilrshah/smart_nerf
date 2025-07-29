# run.py - The Master Pipeline Controller

import os
import subprocess
import argparse
import shutil

def run_command(command):
    """Helper function to run a command and check for errors."""
    print(f"\n> Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
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

def pre_flight_checks():
    """Check if necessary compiled files and directories exist."""
    print("--- Running Pre-flight Checks ---")
    checks_passed = True
    
    # Check if Instant-NGP was compiled
    if not os.path.exists("./vendor/instant-ngp/build/testbed"):
        print("❌ ERROR: Instant-NGP testbed not found.")
        print("  Please compile Instant-NGP by following its installation instructions inside 'vendor/instant-ngp'.")
        checks_passed = False

    # Check for a key Neuralangelo script (assumes it will be there after setup)
    if not os.path.exists("./vendor/neuralangelo/projects/neuralangelo/train.py"):
        print("❌ ERROR: Neuralangelo training script not found.")
        print("  Please ensure the Neuralangelo submodule is correctly cloned in 'vendor/neuralangelo'.")
        checks_passed = False
        
    if checks_passed:
        print("✅ All checks passed.")
    else:
        exit()

def main(args):
    """Main function to orchestrate the pipeline."""
    
    pre_flight_checks()
    
    dataset_source_path = f"data/{args.dataset}"
    if not os.path.exists(dataset_source_path):
        print(f"❌ ERROR: Dataset source directory not found at '{dataset_source_path}'.")
        exit()

    # Define the final optimized JSON path that the pipeline will produce
    results_dir = f"results/{args.dataset}"
    os.makedirs(results_dir, exist_ok=True)
    final_optimized_json = os.path.join(results_dir, "optimized_transforms.json")

    # --- FULL PIPELINE MODE: DATASET OPTIMIZATION ---
    if args.mode == "full":
        print("\n" + "="*50)
        print("    STARTING: Full Intelligent Capture Pipeline")
        print("="*50)

        # --- Step 1: Create Initial Sparse Subset ---
        print("\n--- STEP 1: Creating Initial Sparse Dataset ---")
        current_run_name = "run_01_geom"
        current_run_path = f"experiments/{args.dataset}/{current_run_name}"
        
        cmd_step1 = [
            "python", "src/1_create_initial_subset.py",
            "--dataset", args.dataset,
            "--num_images", str(args.initial_images),
            "--run_name", current_run_name
        ]
        run_command(cmd_step1)
        
        # --- Steps 2 & 3: Iterative Analysis and Augmentation Loop ---
        for i in range(1, args.max_iterations + 1):
            print(f"\n--- ITERATION {i}/{args.max_iterations}: Analysis & Augmentation ---")
            
            # Determine if we are in 'geometry' or 'detail' phase
            if i < args.detail_start_iteration:
                analysis_mode = "geometry"
                n_steps = args.steps_geometry
            else:
                analysis_mode = "detail"
                n_steps = args.steps_detail
            print(f"Current analysis mode: {analysis_mode.upper()}")

            # Step 2: Analyze weakness
            cmd_step2 = [
                "python", "src/2_analyze_weakness.py",
                "--run_path", current_run_path,
                "--mode", analysis_mode,
                "--n_steps", str(n_steps)
            ]
            run_command(cmd_step2)

            # Check if we should stop iterating
            recommendations_file = os.path.join(current_run_path, "recommended_images.txt")
            if not os.path.exists(recommendations_file) or os.path.getsize(recommendations_file) == 0:
                print("\n✅ Analysis complete. No more images recommended. Exiting optimization loop.")
                break
            
            # If this was the last iteration, don't create the next folder
            if i == args.max_iterations:
                print("\nReached max iterations. Exiting optimization loop.")
                break

            # Step 3: Augment dataset for the next run
            next_mode = "detail" if (i + 1) >= args.detail_start_iteration else "geom"
            next_run_name = f"run_{i+1:02d}_{next_mode}"
            
            cmd_step3 = [
                "python", "src/3_augment_dataset.py",
                "--previous_run", current_run_path,
                "--new_run_name", next_run_name
            ]
            run_command(cmd_step3)
            
            current_run_path = f"experiments/{args.dataset}/{next_run_name}"
        
        # --- Post-Loop: Finalize the optimized set ---
        print("\n--- OPTIMIZATION COMPLETE ---")
        print(f"The best dataset was generated in: '{current_run_path}'")
        
        source_json = os.path.join(current_run_path, "transforms.json")
        shutil.copy(source_json, final_optimized_json)
        print(f"✅ Copied final optimized dataset to '{final_optimized_json}'")

    # --- MESHING MODE: Can be run as part of 'full' or standalone ---
    if "mesh" in args.mode:
        print("\n" + "="*50)
        print("    STARTING: 3D Mesh Generation with Neuralangelo")
        print("="*50)
        
        if not os.path.exists(final_optimized_json):
            print(f"❌ ERROR: Cannot run meshing. Optimized file not found at '{final_optimized_json}'.")
            print("  Please run the 'full' pipeline first, or place the file there manually.")
            exit()
            
        # --- Step 4: Run Neuralangelo ---
        print("\n--- STEP 4: Launching Neuralangelo Pipeline ---")
        cmd_step4 = [
            "python", "src/4_run_neuralangelo.py",
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
    parser.add_argument("--mode", choices=['full', 'mesh_only'], default='full', 
                        help="'full': Run optimization and meshing. 'mesh_only': Run only Neuralangelo on an existing optimized set.")
    
    # --- Optimization Loop Parameters ---
    parser.add_argument("--initial_images", type=int, default=15, help="Number of images to start the optimization with.")
    parser.add_argument("--max_iterations", type=int, default=4, help="Maximum number of improvement iterations.")
    parser.add_argument("--detail_start_iteration", type=int, default=2, 
                        help="At which iteration to switch from 'geometry' to 'detail' analysis.")
    parser.add_argument("--steps_geometry", type=int, default=3500, help="Training steps for the geometry analysis phase.")
    parser.add_argument("--steps_detail", type=int, default=15000, help="Training steps for the detail analysis phase.")
    
    args = parser.parse_args()
    main(args)