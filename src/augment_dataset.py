# src/augment_dataset.py - Enhanced with weakness-based prioritization

import json
import os
import argparse
from pathlib import Path
import numpy as np

def load_weakness_metadata(previous_run_path):
    """Load detailed weakness analysis metadata if available."""
    metadata_path = os.path.join(previous_run_path, "analysis", "recommendation_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

def select_best_recommendations(recommendations_data, max_recommendations, selection_strategy="top_priority"):
    """
    Select the best recommendations based on different strategies.
    
    Args:
        recommendations_data: List of recommendation dictionaries with scores
        max_recommendations: Maximum number to select
        selection_strategy: "top_priority", "balanced", or "diverse"
    """
    if not recommendations_data or len(recommendations_data) <= max_recommendations:
        return recommendations_data
    
    if selection_strategy == "top_priority":
        # Simply take the top-scoring recommendations
        return recommendations_data[:max_recommendations]
    
    elif selection_strategy == "balanced":
        # Balance between high priority and diversity
        # Take top 70% from highest priority, 30% from spread
        high_priority_count = int(max_recommendations * 0.7)
        diverse_count = max_recommendations - high_priority_count
        
        selected = recommendations_data[:high_priority_count]
        
        # Add diverse selections from remaining
        remaining = recommendations_data[high_priority_count:]
        if remaining and diverse_count > 0:
            step = len(remaining) // diverse_count
            for i in range(0, len(remaining), max(1, step)):
                if len(selected) < max_recommendations and i < len(remaining):
                    selected.append(remaining[i])
        
        return selected
    
    elif selection_strategy == "diverse":
        # Select recommendations that are spread across the priority range
        step = len(recommendations_data) // max_recommendations
        return [recommendations_data[i] for i in range(0, len(recommendations_data), max(1, step))[:max_recommendations]]
    
    return recommendations_data[:max_recommendations]

def augment_dataset(previous_run_path, new_run_name, full_dataset_json_path, max_recommendations=None, selection_strategy="balanced"):
    print(f"--- Augmenting dataset from run: {previous_run_path} ---")
    print(f"--- Selection strategy: {selection_strategy} ---")

    recommendations_path = os.path.join(previous_run_path, "recommended_images.txt")
    if not os.path.exists(recommendations_path):
        print(f"Error: No recommendations file found at {recommendations_path}.")
        return

    dataset_name = Path(previous_run_path).parts[1]
    new_run_dir = os.path.join("experiments", dataset_name, new_run_name)
    os.makedirs(new_run_dir, exist_ok=True)
    print(f"Created new experiment directory: {new_run_dir}")

    # Load all necessary data
    with open(os.path.join(previous_run_path, "transforms.json"), 'r') as f:
        previous_data = json.load(f)
    with open(full_dataset_json_path, 'r') as f:
        full_data = json.load(f)

    # Load weakness metadata for intelligent selection
    weakness_metadata = load_weakness_metadata(previous_run_path)
    
    if weakness_metadata and 'recommendations' in weakness_metadata:
        print("✅ Found detailed weakness analysis metadata")
        print(f"Analysis mode: {weakness_metadata.get('mode', 'unknown')}")
        print(f"Weak viewpoints identified: {weakness_metadata.get('n_weak_views', 0)}")
        
        recommendations_data = weakness_metadata['recommendations']
        
        # Apply intelligent selection
        if max_recommendations:
            selected_recommendations = select_best_recommendations(
                recommendations_data, max_recommendations, selection_strategy
            )
            print(f"Selected {len(selected_recommendations)} best recommendations using '{selection_strategy}' strategy")
        else:
            selected_recommendations = recommendations_data
            
        recommended_paths = [rec['path'] for rec in selected_recommendations]
        
        # Print selection summary
        if selected_recommendations:
            print("\n--- Selection Summary ---")
            print(f"Top 3 selected recommendations:")
            for i, rec in enumerate(selected_recommendations[:3], 1):
                components = rec.get('weakness_components', {})
                top_weakness = max(components.items(), key=lambda x: x[1]) if components else ('unknown', 0)
                print(f"{i}. Priority: {rec.get('priority_score', 0):.3f}, "
                      f"Distance: {rec.get('distance', 0):.2f}, "
                      f"Main weakness: {top_weakness[0]} ({top_weakness[1]:.3f})")
    else:
        print("⚠️  No detailed metadata found, using simple file-based selection")
        with open(recommendations_path, 'r') as f:
            all_recommended_paths = [line.strip() for line in f if line.strip()]
        
        if max_recommendations and len(all_recommended_paths) > max_recommendations:
            recommended_paths = all_recommended_paths[:max_recommendations]
            print(f"Limited to first {max_recommendations} recommendations")
        else:
            recommended_paths = all_recommended_paths

    print(f"Processing {len(recommended_paths)} recommended images...")
    
    # Debug information
    print(f"\n--- [DEBUG] Dataset Augmentation ---")
    print(f"Previous dataset frames: {len(previous_data['frames'])}")
    print(f"Recommendations to process: {len(recommended_paths)}")
    
    # Start with existing frames, keyed by absolute path
    current_frames_map = {frame['file_path']: frame for frame in previous_data['frames']}

    # Create lookup map from full dataset
    source_base_dir = Path(full_dataset_json_path).parent
    absolute_path_map = {}
    for frame in full_data['frames']:
        abs_path = str((source_base_dir / Path(frame['file_path'])).resolve())
        absolute_path_map[abs_path] = frame

    print(f"Full dataset lookup map has {len(absolute_path_map)} entries")

    # Add recommended frames
    added_count = 0
    skipped_existing = 0
    not_found_count = 0
    
    for path_to_add in recommended_paths:
        if path_to_add not in current_frames_map:
            if path_to_add in absolute_path_map:
                new_frame_data = absolute_path_map[path_to_add].copy()
                new_frame_data['file_path'] = path_to_add
                current_frames_map[path_to_add] = new_frame_data
                added_count += 1
            else:
                print(f"⚠️  WARNING: Recommended path not found in dataset: {Path(path_to_add).name}")
                not_found_count += 1
        else:
            skipped_existing += 1

    final_frames = list(current_frames_map.values())
    
    print(f"\n--- Augmentation Results ---")
    print(f"✅ Added: {added_count} new frames")
    print(f"⏭️  Skipped (already in dataset): {skipped_existing}")
    print(f"❌ Not found: {not_found_count}")
    print(f"📊 Final dataset size: {len(final_frames)} frames")

    # Create final dataset
    subset_data = {
        'camera_angle_x': full_data.get('camera_angle_x'),
        'frames': final_frames
    }
    
    # Add metadata about the augmentation
    if weakness_metadata:
        subset_data['_augmentation_metadata'] = {
            'previous_run': previous_run_path,
            'analysis_mode': weakness_metadata.get('mode'),
            'selection_strategy': selection_strategy,
            'images_added': added_count,
            'total_weak_views': weakness_metadata.get('n_weak_views', 0)
        }

    output_json_path = os.path.join(new_run_dir, "transforms.json")
    with open(output_json_path, 'w') as f:
        json.dump(subset_data, f, indent=4)
        
    print(f"✅ Successfully created augmented dataset at: {output_json_path}")
    print("--- Augmentation complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment a dataset with recommended images based on weakness analysis.")
    parser.add_argument("--previous_run", required=True, help="Path to previous run directory")
    parser.add_argument("--new_run_name", required=True, help="Name for new run directory")
    parser.add_argument("--source", default="data/{dataset}/transforms_train.json", help="Source dataset JSON path")
    parser.add_argument("--max_recommendations", type=int, help="Maximum number of recommendations to add (default: use all)")
    parser.add_argument("--selection_strategy", choices=["top_priority", "balanced", "diverse"], 
                       default="balanced", help="Strategy for selecting recommendations")
    
    args = parser.parse_args()
    
    dataset_name = Path(args.previous_run).parts[1]
    full_json_path = args.source.format(dataset=dataset_name)

    augment_dataset(args.previous_run, args.new_run_name, full_json_path, 
                   args.max_recommendations, args.selection_strategy)