# run.py - The Master Pipeline Controller (Your Superior Architecture)

import os
import subprocess
import argparse
import shutil
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
        if not any(os.path.exists(p) for p in paths_to_check):
            print(f"❌ ERROR: Instant-NGP executable ('{exe_name}') not found for analysis.")
            checks_passed = False

    if mode in ['full', 'mesh_only']:
        # This can be expanded when you reintegrate Neuralangelo
        pass
        
    if checks_passed:
        print("✅ All checks passed.")
    else:
        exit()

def main(args):
    """Main function to orchestrate the pipeline with your two-phase logic."""
    
    pre_flight_checks(args.mode)
    
    dataset_source_path = f"data/{args.dataset}"
    if not os.path.exists(dataset_source_path):
        print(f"❌ ERROR: Dataset source directory not found at '{dataset_source_path}'."); exit()

    results_dir = f"results/{args.dataset}"; os.makedirs(results_dir, exist_ok=True)
    
    if args.mode in ['full', 'optimize_only']:
        print("\n" + "="*50); print("    STARTING: Intelligent Dataset Optimization"); print("="*50)
        
        print("\n--- STEP 1: Creating Initial Sparse Dataset ---")
        current_run_name = "run_01_geom"
        current_run_path = f"experiments/{args.dataset}/{current_run_name}"
        run_command(["python", "src/create_subset.py", "--dataset", args.dataset, "--num_images", str(args.initial_images), "--run_name", current_run_name])

        # --- PHASE 1: Geometry Convergence Loop ---
        print("\n" + "="*50); print("    PHASE 1: GEOMETRY CONVERGENCE"); print("="*50)
        
        geom_iteration = 1
        while True:
            print(f"\n--- GEOMETRY ITERATION {geom_iteration}/{args.max_geom_iterations} ---")

            scene_json_path = os.path.abspath(os.path.join(current_run_path, "transforms.json"))
            snapshot_path = os.path.abspath(os.path.join(current_run_path, "scout_model/scout.ingp"))
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            
            cmd_train = ["python", os.path.abspath("./vendor/instant-ngp/scripts/run.py"), "--scene", scene_json_path, "--save_snapshot", snapshot_path, "--n_steps", "3500"]
            run_command(cmd_train)

            cmd_analyze = [
                "python", "src/analyze_weakness.py",
                "--run_path", current_run_path,
                "--mode", "geometry",
                "--max_recommendations", str(args.max_recommendations)
            ]
            run_command(cmd_analyze)

            recommendations_file = os.path.join(current_run_path, "recommended_images.txt")
            if not os.path.exists(recommendations_file) or os.path.getsize(recommendations_file) == 0:
                print("\n✅ Geometry converged! No more geometric weaknesses found.")
                break
            
            if geom_iteration >= args.max_geom_iterations:
                print(f"\nReached max geometry iterations ({args.max_geom_iterations}). Moving to detail phase.")
                break
            
            geom_iteration += 1
            next_run_name = f"run_{geom_iteration:02d}_geom"
            run_command(["python", "src/augment_dataset.py", "--previous_run", current_run_path, "--new_run_name", next_run_name])
            current_run_path = f"experiments/{args.dataset}/{next_run_name}"

        # --- PHASE 2: Detail Refinement Loop ---
        print("\n" + "="*50); print("    PHASE 2: DETAIL REFINEMENT"); print("="*50)

        for detail_iteration in range(1, args.max_detail_iterations + 1):
            print(f"\n--- DETAIL ITERATION {detail_iteration}/{args.max_detail_iterations} ---")
            
            scene_json_path = os.path.abspath(os.path.join(current_run_path, "transforms.json"))
            snapshot_path = os.path.abspath(os.path.join(current_run_path, "scout_model/scout.ingp"))
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            
            # Train for longer in detail mode to give it a chance to learn sharpness
            cmd_train = ["python", os.path.abspath("./vendor/instant-ngp/scripts/run.py"), "--scene", scene_json_path, "--save_snapshot", snapshot_path, "--n_steps", "5000"]
            run_command(cmd_train)

            cmd_analyze = [
                "python", "src/analyze_weakness.py",
                "--run_path", current_run_path,
                "--mode", "detail",
                "--max_recommendations", str(args.max_recommendations)
            ]
            run_command(cmd_analyze)
            
            recommendations_file = os.path.join(current_run_path, "recommended_images.txt")
            if not os.path.exists(recommendations_file) or os.path.getsize(recommendations_file) == 0:
                print("\n✅ Detail refined! No more recommendations found.")
                break

            # If it's the last detail iteration, we don't need to augment again
            if detail_iteration >= args.max_detail_iterations:
                print("\nReached max detail iterations.")
                break

            next_run_name = f"run_{geom_iteration + detail_iteration:02d}_detail"
            run_command(["python", "src/augment_dataset.py", "--previous_run", current_run_path, "--new_run_name", next_run_name])
            current_run_path = f"experiments/{args.dataset}/{next_run_name}"
            
        print("\n--- OPTIMIZATION COMPLETE ---")
        final_optimized_json = os.path.join("results", args.dataset, "optimized_transforms.json")
        os.makedirs(os.path.dirname(final_optimized_json), exist_ok=True)
        shutil.copy(os.path.join(current_run_path, "transforms.json"), final_optimized_json)
        print(f"✅ Copied final optimized dataset from '{current_run_path}' to '{final_optimized_json}'")
        
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
    
    print("\n" + "="*50); print("      🎉 Pipeline Complete! 🎉"); print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master pipeline controller for the Intelligent 3D Capture project.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=['full', 'optimize_only', 'mesh_only'], default='full')
    parser.add_argument("--initial_images", type=int, default=15)
    
    # New, more logical iteration controls based on your design
    parser.add_argument("--max_geom_iterations", type=int, default=4, help="Max loops for fixing holes.")
    parser.add_argument("--max_detail_iterations", type=int, default=4, help="Max loops for sharpening details after geometry is solid.")
    parser.add_argument("--max_recommendations", type=int, default=10, help="Max new images to add per iteration.")

    args = parser.parse_args()
    main(args)