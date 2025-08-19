# src/manual_exporter.py - Simplified mesh exporter

import sys
from pathlib import Path
import tyro
import torch
import numpy as np

# Patch torch.load
original_torch_load = torch.load
torch.load = lambda f, map_location=None, **kwargs: original_torch_load(f, map_location=map_location, weights_only=False)

try:
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
except ImportError as e:
    print(f"❌ Import error: {e}", file=sys.stderr)
    sys.exit(1)

def get_density_function(model):
    """Extract density function from model."""
    field = model.field
    device = next(model.parameters()).device
    
    def density_fn(positions: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            positions = positions.to(device)
            
            # Try different methods to get density
            try:
                if hasattr(field, 'density_fn'):
                    density = field.density_fn(positions)
                elif hasattr(field, 'get_density'):
                    from nerfstudio.cameras.rays import RaySamples
                    # Create minimal ray samples
                    ray_samples = RaySamples(
                        frustums=type('Frustums', (), {
                            'get_positions': lambda: positions,
                            'shape': positions.shape[:-1]
                        })(),
                        spacing_starts=torch.zeros_like(positions[..., :1]),
                        spacing_ends=torch.ones_like(positions[..., :1]),
                        spacing_to_euclidean_fn=lambda x: x,
                    )
                    density = field.get_density(ray_samples)
                else:
                    raise ValueError(f"Cannot extract density from field type {type(field)}")
                
                # Handle different return types
                if isinstance(density, tuple):
                    density = density[0]
                if isinstance(density, dict):
                    density = density.get('density', torch.zeros_like(positions[..., 0]))
                
                # Ensure proper shape
                if density.dim() > 1 and density.shape[-1] == 1:
                    density = density.squeeze(-1)
                
                return torch.clamp(density, 0, 100)  # Reasonable clamp
                
            except Exception as e:
                print(f"Density extraction failed: {e}")
                # Return zero density as fallback
                return torch.zeros(positions.shape[0], device=device)
    
    return density_fn

def main(config_path: Path, output_path: Path, resolution: int = 512):
    """Simple mesh export."""
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    print(f"Loading model from: {config_path}")
    print(f"Output path: {output_path}")
    print(f"Resolution: {resolution}")
    
    try:
        # Load pipeline
        _, pipeline, _, _ = eval_setup(config_path, test_mode="inference")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipeline.to(device)
        pipeline.eval()
        print("✅ Pipeline loaded")
        
        # Get scene bounds
        model = pipeline.model
        if hasattr(model, 'scene_box') and hasattr(model.scene_box, 'aabb'):
            aabb = model.scene_box.aabb
            bbox_min = aabb[0].cpu().numpy()
            bbox_max = aabb[1].cpu().numpy()
        else:
            # Default bounds
            bbox_min = np.array([-2.0, -2.0, -2.0])
            bbox_max = np.array([2.0, 2.0, 2.0])
        
        print(f"Scene bounds: {bbox_min} to {bbox_max}")
        
        # Get density function
        density_fn = get_density_function(model)
        
        # Test different thresholds
        thresholds = [0.5, 0.3, 0.1, 0.05, 0.01]
        
        for threshold in thresholds:
            print(f"\nTrying threshold: {threshold}")
            try:
                mesh = generate_mesh_with_multires_marching_cubes(
                    geometry_callable_field=density_fn,
                    resolution=resolution,
                    bounding_box_min=tuple(bbox_min),
                    bounding_box_max=tuple(bbox_max),
                    isosurface_threshold=threshold,
                )
                
                if hasattr(mesh, 'vertices') and len(mesh.vertices) > 100:
                    vertex_count = len(mesh.vertices)
                    print(f"✅ Generated mesh with {vertex_count:,} vertices")
                    
                    # Save mesh
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    mesh.export(str(output_path))
                    
                    if output_path.exists():
                        file_size = output_path.stat().st_size
                        print(f"✅ Mesh saved: {file_size / 1024 / 1024:.1f} MB")
                        return
                    else:
                        print("❌ Failed to save mesh file")
                else:
                    print(f"⚠️ Mesh too small: {len(mesh.vertices) if hasattr(mesh, 'vertices') else 0} vertices")
                    
            except Exception as e:
                print(f"❌ Failed with threshold {threshold}: {e}")
        
        print("\n❌ All thresholds failed")
        print("💡 Suggestions:")
        print("   1. Check if model trained properly")
        print("   2. Try: ns-export poisson --load-config", config_path)
        print("   3. Use lower resolution (256 or 128)")
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)
    
    finally:
        torch.load = original_torch_load

if __name__ == "__main__":
    tyro.cli(main)