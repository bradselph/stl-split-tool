import trimesh
from trimesh import repair, intersections
import numpy as np
import sys
import os
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def validate_mesh(mesh: trimesh.Trimesh) -> Tuple[bool, str]:
    if mesh.is_empty:
        return False, "Error: Mesh is empty."
    if len(mesh.faces) < 4:
        return False, "Error: Mesh has too few faces to split."
    if not mesh.is_watertight:
        return False, "Warning: Mesh is not watertight. Split may produce incomplete halves."
    if hasattr(mesh, "is_winding_consistent") and not mesh.is_winding_consistent:
        return False, "Warning: Mesh has inconsistent winding. Results may be unreliable."
    return True, "Mesh validation passed."

def find_symmetry_axis(mesh: trimesh.Trimesh) -> Tuple[str, int, float]:
    axes = ['x', 'y', 'z']
    scores = []
    vertices = mesh.vertices - mesh.centroid
    
    for i in range(3):
        v = vertices[:, i]
        pos = np.abs(v[v >= 0.0])
        neg = np.abs(v[v < 0.0])
        
        if pos.size > 0 and neg.size > 0:
            score = 1.0 - abs(np.mean(pos) - np.mean(neg)) / (np.mean(pos) + np.mean(neg) + 1e-12)
        else:
            score = 0.0
        scores.append(score)
    
    idx = int(np.argmax(scores))
    return axes[idx], idx, float(scores[idx])

def find_optimal_split_plane(mesh: trimesh.Trimesh, axis_idx: int) -> np.ndarray:
    coords = mesh.vertices[:, axis_idx]
    candidates = [
        np.median(coords),
        np.mean(coords),
        mesh.centroid[axis_idx],
        (np.min(coords) + np.max(coords)) / 2.0
    ]
    
    best_pos = candidates[0]
    best_score = float('inf')
    
    for c in candidates:
        above = np.sum(coords >= c)
        below = np.sum(coords < c)
        imbalance = abs(above - below)
        if imbalance < best_score:
            best_score = imbalance
            best_pos = c
    
    origin = mesh.centroid.copy()
    origin[axis_idx] = best_pos
    return origin

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def seal_mesh_robustly(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if mesh is None or mesh.is_empty:
        return mesh
    try:
        repair.fill_holes(mesh)
    except Exception as e:
        print(f"  fill_holes failed: {e}")
    if hasattr(repair, "fix_normals"):
        try:
            repair.fix_normals(mesh)
        except Exception:
            pass
    if hasattr(repair, "fix_inversion"):
        try:
            repair.fix_inversion(mesh)
        except Exception:
            pass
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    if not mesh.is_watertight:
        try:
            edges = mesh.edges_unique
            edge_face_count = mesh.edges_unique_inverse
            edge_counts = np.bincount(edge_face_count)
            boundary_edges = edges[edge_counts == 1]
            
            if len(boundary_edges) > 0:
                print(f"  Detected {len(boundary_edges)} boundary edges, attempting seal...")
                boundary_verts_idx = np.unique(boundary_edges.flatten())
                boundary_verts = mesh.vertices[boundary_verts_idx]
                if len(boundary_verts) < 100:
                    try:
                        center = np.mean(boundary_verts, axis=0)
                        centered = boundary_verts - center
                        _, _, vh = np.linalg.svd(centered, full_matrices=False)
                        normal = vh[2]
                        
                        basis1 = vh[0]
                        basis2 = vh[1]
                        coords_2d = np.column_stack([
                            np.dot(centered, basis1),
                            np.dot(centered, basis2)
                        ])
                        
                        from scipy.spatial import Delaunay
                        tri = Delaunay(coords_2d)
                        new_faces = boundary_verts_idx[tri.simplices]
                        mesh.faces = np.vstack([mesh.faces, new_faces])
                        
                        print("  Successfully filled hole with planar triangulation")
                    except Exception as e:
                        print(f"  Planar hole filling failed: {e}")
        except Exception as e:
            print(f"  Manual hole filling failed: {e}")
    
    return mesh

def create_watertight_half(mesh: trimesh.Trimesh, plane_origin: np.ndarray, plane_normal: np.ndarray, cap: bool = True) -> Optional[trimesh.Trimesh]:
    try:
        half = intersections.slice_mesh_plane(
            mesh=mesh,
            plane_normal=plane_normal,
            plane_origin=plane_origin,
            cap=cap
        )
        
        if half is None or len(half.faces) == 0:
            return None
        
        if cap:
            print(f"  Sealing half (normal: [{plane_normal[0]:.2f}, {plane_normal[1]:.2f}, {plane_normal[2]:.2f}])...")
            half = seal_mesh_robustly(half)
        
        return half if len(half.faces) > 0 else None
        
    except Exception as e:
        print(f"  Error creating half: {e}")
        return None

def split_mesh_watertight(mesh: trimesh.Trimesh, plane_origin: np.ndarray, plane_normal: np.ndarray) -> Optional[Tuple[trimesh.Trimesh, trimesh.Trimesh]]:
    n = normalize(np.asarray(plane_normal, dtype=float))
    
    print("\nAttempting split with capping...")
    half_A = create_watertight_half(mesh, plane_origin, n, cap=True)
    half_B = create_watertight_half(mesh, plane_origin, -n, cap=True)
    
    if half_A and half_B and not half_A.is_empty and not half_B.is_empty:
        if half_A.is_watertight and half_B.is_watertight:
            print("Split successful: both halves are watertight")
            return half_A, half_B
        else:
            print("Split succeeded but halves need additional sealing")
            if not half_A.is_watertight:
                print("  Additional sealing for half A...")
                half_A = seal_mesh_robustly(half_A)
            if not half_B.is_watertight:
                print("  Additional sealing for half B...")
                half_B = seal_mesh_robustly(half_B)
            return half_A, half_B
    
    print("Capped split failed, trying without cap...")
    half_A = create_watertight_half(mesh, plane_origin, n, cap=False)
    half_B = create_watertight_half(mesh, plane_origin, -n, cap=False)
    
    if half_A and half_B and not half_A.is_empty and not half_B.is_empty:
        print("Split without cap succeeded, applying sealing...")
        half_A = seal_mesh_robustly(half_A)
        half_B = seal_mesh_robustly(half_B)
        return half_A, half_B
    
    print("Trying offset strategies...")
    bbox = mesh.bounds
    size = bbox[1] - bbox[0]
    eps = max(np.min(size) * 1e-6, 1e-6)
    
    for factor in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        origin_offset = plane_origin + n * (eps * factor)
        half_A = create_watertight_half(mesh, origin_offset, n, cap=True)
        half_B = create_watertight_half(mesh, origin_offset, -n, cap=True)
        
        if half_A and half_B and not half_A.is_empty and not half_B.is_empty:
            print(f"Split successful with positive offset (eps * {factor})")
            if not half_A.is_watertight:
                half_A = seal_mesh_robustly(half_A)
            if not half_B.is_watertight:
                half_B = seal_mesh_robustly(half_B)
            return half_A, half_B
        
        origin_offset = plane_origin - n * (eps * factor)
        half_A = create_watertight_half(mesh, origin_offset, n, cap=True)
        half_B = create_watertight_half(mesh, origin_offset, -n, cap=True)
        
        if half_A and half_B and not half_A.is_empty and not half_B.is_empty:
            print(f"Split successful with negative offset (eps * {factor})")
            if not half_A.is_watertight:
                half_A = seal_mesh_robustly(half_A)
            if not half_B.is_watertight:
                half_B = seal_mesh_robustly(half_B)
            return half_A, half_B
    
    return None

def visualize_mesh_with_plane(mesh: trimesh.Trimesh, axis_idx: int, split_pos: float):
    try:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        sample_rate = max(1, len(faces) // 3000)
        sampled_faces = faces[::sample_rate]
        
        mesh_collection = Poly3DCollection(vertices[sampled_faces], alpha=0.25, facecolor='cyan', edgecolor='gray', linewidth=0.1)
        ax.add_collection3d(mesh_collection)
        
        bbox_min, bbox_max = mesh.bounds
        
        if axis_idx == 0:
            yy, zz = np.meshgrid(
                np.linspace(bbox_min[1], bbox_max[1], 2),
                np.linspace(bbox_min[2], bbox_max[2], 2)
            )
            xx = np.full_like(yy, split_pos)
        elif axis_idx == 1:
            xx, zz = np.meshgrid(
                np.linspace(bbox_min[0], bbox_max[0], 2),
                np.linspace(bbox_min[2], bbox_max[2], 2)
            )
            yy = np.full_like(xx, split_pos)
        else:
            xx, yy = np.meshgrid(
                np.linspace(bbox_min[0], bbox_max[0], 2),
                np.linspace(bbox_min[1], bbox_max[1], 2)
            )
            zz = np.full_like(xx, split_pos)
        
        ax.plot_surface(xx, yy, zz, alpha=0.6, color='red')
        
        margin = (bbox_max - bbox_min) * 0.1
        ax.set_xlim([bbox_min[0] - margin[0], bbox_max[0] + margin[0]])
        ax.set_ylim([bbox_min[1] - margin[1], bbox_max[1] + margin[1]])
        ax.set_zlim([bbox_min[2] - margin[2], bbox_max[2] + margin[2]])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        axis_names = ['X', 'Y', 'Z']
        ax.set_title(f'Split Plane Preview: {axis_names[axis_idx]} = {split_pos:.4f}')
        
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        
    except Exception as e:
        print(f"Visualization error: {e}")

def auto_cut_stl(filename: str, output_dir: Optional[str] = None, manual_axis_idx: Optional[int] = None, manual_plane_origin: Optional[np.ndarray] = None):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Input file not found: {filename}")
    
    if not filename.lower().endswith('.stl'):
        raise ValueError("Input must be STL file")
    
    mesh = trimesh.load(filename, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("File is not a valid mesh")
    
    print(f"Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    is_valid, message = validate_mesh(mesh)
    print(message)
    if not is_valid:
        raise RuntimeError(message)
    
    if manual_axis_idx is not None and manual_plane_origin is not None:
        axis_idx = manual_axis_idx
        plane_origin = manual_plane_origin
        axes = ['X', 'Y', 'Z']
        print(f"Manual mode: {axes[axis_idx]} axis at {plane_origin[axis_idx]:.4f}")
    else:
        best_axis, axis_idx, confidence = find_symmetry_axis(mesh)
        print(f"Auto-detected: {best_axis.upper()} (confidence: {confidence:.3f})")
        
        if confidence < 0.25:
            print("Warning: Low confidence")
        
        plane_origin = find_optimal_split_plane(mesh, axis_idx)
    
    plane_normal = np.zeros(3, dtype=float)
    plane_normal[axis_idx] = 1.0
    plane_normal = normalize(plane_normal)
    
    print(f"Plane origin: [{plane_origin[0]:.4f}, {plane_origin[1]:.4f}, {plane_origin[2]:.4f}]")
    print(f"Plane normal: [{plane_normal[0]:.4f}, {plane_normal[1]:.4f}, {plane_normal[2]:.4f}]")
    
    result = split_mesh_watertight(mesh, plane_origin, plane_normal)
    
    if result is None:
        print("\nAll split strategies failed, attempting final fallback...")
        
        try:
            half_A = intersections.slice_mesh_plane(mesh, plane_normal, plane_origin, cap=False)
            half_B = intersections.slice_mesh_plane(mesh, -plane_normal, plane_origin, cap=False)
            
            if half_A and half_B and len(half_A.faces) > 0 and len(half_B.faces) > 0:
                print("Fallback produced uncapped halves, applying aggressive sealing...")
                half_A = seal_mesh_robustly(half_A)
                half_B = seal_mesh_robustly(half_B)
                result = (half_A, half_B)
            else:
                raise RuntimeError("Fallback produced no valid halves")
        except Exception as e:
            raise RuntimeError(f"Split failed: {e}")
    
    half_A, half_B = result
    
    watertight_A = bool(getattr(half_A, "is_watertight", False))
    watertight_B = bool(getattr(half_B, "is_watertight", False))
    
    print(f"\nHalf A: {len(half_A.vertices)} verts, {len(half_A.faces)} faces, watertight: {watertight_A}")
    print(f"Half B: {len(half_B.vertices)} verts, {len(half_B.faces)} faces, watertight: {watertight_B}")
    
    if not (watertight_A and watertight_B):
        print("\nWarning: One or both halves are not watertight.")
        print("Recommendation: Import into mesh repair software (e.g., Meshmixer, Netfabb) before 3D printing.")
    else:
        print("\nSuccess: Both halves are watertight and ready for 3D printing.")
    
    base = os.path.splitext(os.path.basename(filename))[0]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        half_A_file = os.path.join(output_dir, f"{base}_half_A.stl")
        half_B_file = os.path.join(output_dir, f"{base}_half_B.stl")
    else:
        half_A_file = f"{base}_half_A.stl"
        half_B_file = f"{base}_half_B.stl"
    
    try:
        half_A.export(half_A_file)
        half_B.export(half_B_file)
    except Exception as e:
        raise RuntimeError(f"Export failed: {e}")
    
    print(f"\nExported:\n  {os.path.abspath(half_A_file)}\n  {os.path.abspath(half_B_file)}")

def get_manual_split_config(mesh: trimesh.Trimesh) -> Tuple[int, np.ndarray]:
    bbox_min, bbox_max = mesh.bounds
    centroid = mesh.centroid
    
    print("\nMesh bounds:")
    for i, ax in enumerate(['X', 'Y', 'Z']):
        print(f"  {ax}: [{bbox_min[i]:.4f} to {bbox_max[i]:.4f}] size: {bbox_max[i]-bbox_min[i]:.4f}")
    print(f"Centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")
    
    while True:
        axis_choice = input("\nSplit axis (x/y/z): ").strip().lower()
        if axis_choice in ['x', 'y', 'z']:
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis_choice]
            break
        print("Invalid")
    
    print(f"\nPosition for {axis_choice.upper()}:")
    print("  1. Centroid")
    print("  2. Bounding box center")
    print("  3. Custom coordinate")
    print("  4. Percentage (0-100%)")
    
    while True:
        pos_choice = input("\nSelect (1-4): ").strip()
        
        if pos_choice == '1':
            position = centroid[axis_idx]
            break
        elif pos_choice == '2':
            position = (bbox_min[axis_idx] + bbox_max[axis_idx]) / 2.0
            break
        elif pos_choice == '3':
            try:
                position = float(input(f"Enter {axis_choice.upper()} value: "))
                break
            except ValueError:
                print("Invalid number")
        elif pos_choice == '4':
            try:
                percent = float(input("Percentage (0-100): "))
                if 0 <= percent <= 100:
                    position = bbox_min[axis_idx] + (bbox_max[axis_idx] - bbox_min[axis_idx]) * (percent / 100.0)
                    break
                print("Must be 0-100")
            except ValueError:
                print("Invalid number")
        else:
            print("Invalid")
    
    plane_origin = centroid.copy()
    plane_origin[axis_idx] = position
    
    print(f"\nSelected: {axis_choice.upper()} = {position:.4f}")
    print(f"Plane origin: [{plane_origin[0]:.4f}, {plane_origin[1]:.4f}, {plane_origin[2]:.4f}]")
    
    print("\nGenerating visualization...")
    visualize_mesh_with_plane(mesh, axis_idx, position)
    input("Press Enter after viewing (close window manually)...")
    plt.close('all')
    
    return axis_idx, plane_origin

def interactive_mode():
    print("=" * 60)
    print("STL Split Tool")
    print("=" * 60)
    
    while True:
        input_file = input("\nSTL file path (or 'q'): ").strip()
        
        if input_file.lower() == 'q':
            sys.exit(0)
        
        if not input_file:
            print("No file specified")
            continue
        
        input_file = input_file.strip("'\"")
        
        if not os.path.isfile(input_file):
            print(f"Not found: {input_file}")
            continue
        
        if not input_file.lower().endswith('.stl'):
            print("Must be .stl file")
            continue
        
        break
    
    use_custom_dir = input("\nCustom output directory? (y/n): ").strip().lower()
    
    output_directory = None
    if use_custom_dir == 'y':
        output_directory = input("Output path: ").strip().strip("'\"")
        if not output_directory:
            output_directory = None
    
    print("\nLoading...")
    try:
        mesh = trimesh.load(input_file, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            print("Error: Invalid mesh")
            sys.exit(1)
    except Exception as e:
        print(f"Load error: {e}")
        sys.exit(1)
    
    print(f"Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    print("\nMode:")
    print("  1. Auto-detect")
    print("  2. Manual")
    
    mode_choice = input("\nSelect (1/2): ").strip()
    
    manual_axis_idx = None
    manual_plane_origin = None
    
    if mode_choice == '2':
        while True:
            manual_axis_idx, manual_plane_origin = get_manual_split_config(mesh)
            
            print("\nConfiguration:")
            print(f"  Input: {input_file}")
            print(f"  Output: {output_directory if output_directory else 'Same directory'}")
            print(f"  Mode: Manual")
            print(f"  Split: {['X', 'Y', 'Z'][manual_axis_idx]} = {manual_plane_origin[manual_axis_idx]:.4f}")
            
            confirm = input("\nProceed with this configuration? (y/n/q): ").strip().lower()
            if confirm == 'y':
                break
            elif confirm == 'q':
                print("Cancelled")
                sys.exit(0)
            else:
                print("\nRestarting manual configuration...")
    else:
        print("\nConfiguration:")
        print(f"  Input: {input_file}")
        print(f"  Output: {output_directory if output_directory else 'Same directory'}")
        print(f"  Mode: Auto")
        
        confirm = input("\nProceed? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled")
            sys.exit(0)
    
    print("\nProcessing...")
    print("-" * 60)
    
    try:
        auto_cut_stl(input_file, output_directory, manual_axis_idx, manual_plane_origin)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        interactive_mode()
    else:
        input_file = sys.argv[1]
        output_directory = sys.argv[2] if len(sys.argv) > 2 else None
        
        try:
            auto_cut_stl(input_file, output_directory)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(2)
            