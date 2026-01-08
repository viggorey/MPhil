"""
Interactive tool to manage branch sets.
Create, update, delete, and view branch sets with branch axis/radius calculation.
"""

import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path to import dlt_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "3D_reconstruction"))
from dlt_utils import reconstruct_3d_point

BRANCH_SETS_FILE = Path(__file__).parent / "branch_sets.json"
CAMERA_ORDER = ["Left", "Top", "Right", "Front"]


def load_branch_sets():
    """Load branch sets from JSON file."""
    if BRANCH_SETS_FILE.exists():
        with open(BRANCH_SETS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_branch_sets(branch_sets):
    """Save branch sets to JSON file."""
    with open(BRANCH_SETS_FILE, 'w') as f:
        json.dump(branch_sets, f, indent=2)


def input_branch_points_2d():
    """Interactively input 2D branch points for each camera."""
    branch_points = {}
    
    print("\nEnter branch axis points for each camera view.")
    print("These are points along the branch centerline in each camera view.")
    
    for camera_name in CAMERA_ORDER:
        print(f"\n--- {camera_name} Camera ---")
        print("Enter points along the branch centerline (u, v).")
        print("Enter 'done' when finished (need at least 2 points):")
        
        points = []
        while True:
            point_str = input(f"  Point {len(points) + 1} (u, v) or 'done': ").strip()
            if point_str.lower() == 'done':
                if len(points) >= 2:
                    break
                else:
                    print(f"    Need at least 2 points. Currently have {len(points)}.")
                    continue
            
            try:
                coords = [float(x.strip()) for x in point_str.split(',')]
                if len(coords) == 2:
                    points.append(coords)
                    print(f"    Added: {coords}")
                else:
                    print("    Error: Need 2 coordinates (u, v)")
            except ValueError:
                print("    Error: Invalid input. Use format: u, v")
        
        branch_points[camera_name] = np.array(points)
    
    return branch_points


def input_surface_points_2d():
    """Interactively input 2D surface points for each camera."""
    surface_points = {}
    
    print("\nEnter branch surface points for each camera view.")
    print("These are point pairs that define the branch surface edges.")
    
    for camera_name in CAMERA_ORDER:
        print(f"\n--- {camera_name} Camera ---")
        print("Enter surface point pairs (u, v).")
        print("Enter 'done' when finished (need at least 2 point pairs):")
        
        points = []
        while True:
            point_str = input(f"  Point pair {len(points) + 1} (u, v) or 'done': ").strip()
            if point_str.lower() == 'done':
                if len(points) >= 2:
                    break
                else:
                    print(f"    Need at least 2 point pairs. Currently have {len(points)}.")
                    continue
            
            try:
                coords = [float(x.strip()) for x in point_str.split(',')]
                if len(coords) == 2:
                    points.append(coords)
                    print(f"    Added: {coords}")
                else:
                    print("    Error: Need 2 coordinates (u, v)")
            except ValueError:
                print("    Error: Invalid input. Use format: u, v")
        
        surface_points[camera_name] = np.array(points)
    
    return surface_points


def reconstruct_branch_axis(A, branch_points_2d):
    """
    Reconstruct 3D branch axis from 2D points in multiple camera views.
    Based on C3_VizBranchAnts.py
    """
    from scipy import stats
    
    def get_direction_from_points(points):
        """Calculate the primary direction of a set of 2D points."""
        x = points[:, 0]
        y = points[:, 1]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        direction = np.array([1.0, slope])
        return direction / np.linalg.norm(direction)
    
    # Filter out empty arrays and get valid camera indices
    valid_cameras = []
    points_list = []
    for i, cam_name in enumerate(CAMERA_ORDER):
        if cam_name in branch_points_2d and branch_points_2d[cam_name].size > 0:
            valid_cameras.append(i)
            points_list.append(branch_points_2d[cam_name])
    
    if len(valid_cameras) < 2:
        raise ValueError("Need at least 2 cameras with points for reconstruction")
    
    # Use only valid cameras and their corresponding coefficients
    A_subset = A[:, valid_cameras]
    n_cameras = len(valid_cameras)
    midline_vectors_2d = np.zeros((n_cameras, 2))
    
    for i, points in enumerate(points_list):
        direction = get_direction_from_points(points)
        midline_vectors_2d[i] = direction
    
    M = np.zeros((2 * n_cameras, 3 + n_cameras))
    
    for i in range(n_cameras):
        L = A_subset[:, i]
        u, v = midline_vectors_2d[i]
        
        row = 2 * i
        M[row, 0] = (L[0] - L[3]*L[8]) / (L[8] + 1)
        M[row, 1] = (L[1] - L[3]*L[9]) / (L[9] + 1)
        M[row, 2] = (L[2] - L[3]*L[10]) / (L[10] + 1)
        M[row, 3 + i] = -u
        
        row = 2 * i + 1
        M[row, 0] = (L[4] - L[7]*L[8]) / (L[8] + 1)
        M[row, 1] = (L[5] - L[7]*L[9]) / (L[9] + 1)
        M[row, 2] = (L[6] - L[7]*L[10]) / (L[10] + 1)
        M[row, 3 + i] = -v
    
    # Solve for axis direction and point
    result = np.linalg.lstsq(M, np.zeros(2 * n_cameras), rcond=1e-10)[0]
    direction = result[:3]
    direction = direction / np.linalg.norm(direction)
    
    # Find a point on the axis by reconstructing the centroid of branch points
    # Use the mean of branch points from first valid camera
    centroid_2d = np.mean(points_list[0], axis=0)
    
    # Build system to find point on axis
    P = np.zeros((2 * n_cameras, 3))
    b = np.zeros(2 * n_cameras)
    
    for i, cam_idx in enumerate(valid_cameras):
        L = A[:, cam_idx]
        # Use mean of points from this camera
        u, v = np.mean(points_list[i], axis=0)
        
        row = 2 * i
        P[row, 0] = u*L[8] - L[0]
        P[row, 1] = u*L[9] - L[1]
        P[row, 2] = u*L[10] - L[2]
        b[row] = L[3] - u
        
        P[row + 1, 0] = v*L[8] - L[4]
        P[row + 1, 1] = v*L[9] - L[5]
        P[row + 1, 2] = v*L[10] - L[6]
        b[row + 1] = L[7] - v
    
    point_on_line = np.linalg.lstsq(P, b, rcond=1e-10)[0]
    
    return direction.tolist(), point_on_line.tolist()


def calculate_branch_radius(A, axis_direction, axis_point, surface_points_2d):
    """Calculate branch radius from surface points."""
    # Reconstruct surface points to 3D
    surface_points_3d = []
    
    # Get valid cameras
    valid_cameras = [i for i, cam_name in enumerate(CAMERA_ORDER) 
                     if cam_name in surface_points_2d and len(surface_points_2d[cam_name]) > 0]
    
    if len(valid_cameras) < 2:
        raise ValueError("Need at least 2 cameras for surface point reconstruction")
    
    A_subset = A[:, valid_cameras]
    
    # Find the minimum number of points across all cameras (to handle mismatched counts)
    n_points_per_camera = [len(surface_points_2d[CAMERA_ORDER[cam_idx]]) for cam_idx in valid_cameras]
    n_points = min(n_points_per_camera) if n_points_per_camera else 0
    
    if n_points < 2:
        raise ValueError("Need at least 2 surface points per camera to calculate radius")
    
    # Reconstruct each surface point (each point is seen from multiple cameras)
    for point_idx in range(n_points):
        point_2d_list = []
        for cam_idx in valid_cameras:
            cam_name = CAMERA_ORDER[cam_idx]
            if point_idx < len(surface_points_2d[cam_name]):
                point_2d_list.append(np.array(surface_points_2d[cam_name][point_idx]))
            else:
                # Skip this point if not available in this camera
                continue
        
        # Need at least 2 cameras to reconstruct
        if len(point_2d_list) >= 2:
            try:
                point_3d = reconstruct_3d_point(A_subset, point_2d_list)
                surface_points_3d.append(point_3d)
            except Exception as e:
                print(f"    Warning: Could not reconstruct surface point {point_idx}: {e}")
                continue
    
    if len(surface_points_3d) < 2:
        raise ValueError("Need at least 2 successfully reconstructed surface points to calculate radius")
    
    # Calculate distances from axis
    axis_direction = np.array(axis_direction)
    axis_point = np.array(axis_point)
    distances = []
    
    for point_3d in surface_points_3d:
        point_3d = np.array(point_3d)
        # Vector from axis point to surface point
        vec = point_3d - axis_point
        # Project onto axis
        proj = np.dot(vec, axis_direction) * axis_direction
        # Perpendicular distance
        perp = vec - proj
        distance = np.linalg.norm(perp)
        distances.append(distance)
    
    # Average radius
    radius = np.mean(distances)
    
    return float(radius)


def create_branch_set():
    """Create a new branch set."""
    print("\n" + "="*60)
    print("CREATE NEW BRANCH SET")
    print("="*60)
    
    # Get branch set name
    name = input("\nEnter branch set name (e.g., '2024-01-15_video1'): ").strip()
    if not name:
        print("Error: Name cannot be empty")
        return
    
    branch_sets = load_branch_sets()
    if name in branch_sets:
        overwrite = input(f"Branch set '{name}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Cancelled.")
            return
    
    # Load calibration sets to get camera coefficients
    cal_sets_file = Path(__file__).parent.parent / "Calibration" / "calibration_sets.json"
    if not cal_sets_file.exists():
        print("Error: No calibration sets found. Please create calibration sets first.")
        return
    
    with open(cal_sets_file, 'r') as f:
        calibration_sets = json.load(f)
    
    print("\nAvailable calibration sets:")
    for i, cal_name in enumerate(calibration_sets.keys(), 1):
        print(f"  {i}. {cal_name}")
    
    cal_name = input("\nEnter calibration set name to use: ").strip()
    if cal_name not in calibration_sets:
        print(f"Error: Calibration set '{cal_name}' not found.")
        return
    
    cal_set = calibration_sets[cal_name]
    A = np.array(cal_set['camera_coefficients'])
    
    # Input branch points
    branch_points_2d = input_branch_points_2d()
    
    # Input surface points
    surface_points_2d = input_surface_points_2d()
    
    # Reconstruct branch axis
    try:
        axis_direction, axis_point = reconstruct_branch_axis(A, branch_points_2d)
        print(f"\n✓ Branch axis reconstructed")
        print(f"  Direction: {axis_direction}")
        print(f"  Point on axis: {axis_point}")
    except Exception as e:
        print(f"Error reconstructing branch axis: {e}")
        return
    
    # Calculate radius
    try:
        radius = calculate_branch_radius(A, axis_direction, axis_point, surface_points_2d)
        print(f"✓ Branch radius calculated: {radius:.4f} mm")
    except Exception as e:
        print(f"Error calculating radius: {e}")
        radius = None
    
    # Convert numpy arrays to lists for JSON
    branch_points_dict = {}
    for cam_name, points in branch_points_2d.items():
        branch_points_dict[cam_name] = points.tolist()
    
    surface_points_dict = {}
    for cam_name, points in surface_points_2d.items():
        surface_points_dict[cam_name] = points.tolist()
    
    # Save branch set
    branch_set = {
        "name": name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "created": datetime.now().isoformat(),
        "calibration_set": cal_name,
        "branch_points_2d": branch_points_dict,
        "surface_points_2d": surface_points_dict,
        "axis_direction": axis_direction,
        "axis_point": axis_point,
        "radius": radius
    }
    
    branch_sets[name] = branch_set
    save_branch_sets(branch_sets)
    
    print(f"\n✓ Branch set '{name}' saved successfully!")


def view_branch_set():
    """View an existing branch set."""
    branch_sets = load_branch_sets()
    
    if not branch_sets:
        print("\nNo branch sets found.")
        return
    
    print("\n" + "="*60)
    print("VIEW BRANCH SET")
    print("="*60)
    print("\nAvailable branch sets:")
    for i, name in enumerate(branch_sets.keys(), 1):
        print(f"  {i}. {name}")
    
    name = input("\nEnter branch set name: ").strip()
    if name not in branch_sets:
        print(f"Error: Branch set '{name}' not found.")
        return
    
    branch_set = branch_sets[name]
    print(f"\n{'='*60}")
    print(f"Branch Set: {name}")
    print(f"{'='*60}")
    print(f"Date: {branch_set.get('date', 'N/A')}")
    print(f"Calibration Set: {branch_set.get('calibration_set', 'N/A')}")
    print(f"Radius: {branch_set.get('radius', 'N/A'):.4f} mm" if branch_set.get('radius') else "Radius: N/A")
    print(f"\nAxis Direction: {branch_set.get('axis_direction', 'N/A')}")
    print(f"Axis Point: {branch_set.get('axis_point', 'N/A')}")


def delete_branch_set():
    """Delete a branch set."""
    branch_sets = load_branch_sets()
    
    if not branch_sets:
        print("\nNo branch sets found.")
        return
    
    print("\n" + "="*60)
    print("DELETE BRANCH SET")
    print("="*60)
    print("\nAvailable branch sets:")
    for i, name in enumerate(branch_sets.keys(), 1):
        print(f"  {i}. {name}")
    
    name = input("\nEnter branch set name to delete: ").strip()
    if name not in branch_sets:
        print(f"Error: Branch set '{name}' not found.")
        return
    
    confirm = input(f"Are you sure you want to delete '{name}'? (yes/no): ").strip().lower()
    if confirm == 'yes':
        del branch_sets[name]
        save_branch_sets(branch_sets)
        print(f"✓ Branch set '{name}' deleted.")
    else:
        print("Cancelled.")


def list_branch_sets():
    """List all branch sets."""
    branch_sets = load_branch_sets()
    
    if not branch_sets:
        print("\nNo branch sets found.")
        return
    
    print("\n" + "="*60)
    print("BRANCH SETS")
    print("="*60)
    print(f"\nTotal: {len(branch_sets)}")
    print("\nName".ljust(30) + "Date".ljust(15) + "Calibration Set".ljust(20) + "Radius")
    print("-" * 80)
    
    for name, branch_set in branch_sets.items():
        date = branch_set.get('date', 'N/A')
        cal_set = branch_set.get('calibration_set', 'N/A')
        radius = branch_set.get('radius', None)
        radius_str = f"{radius:.4f} mm" if radius else "N/A"
        print(f"{name.ljust(30)}{date.ljust(15)}{cal_set.ljust(20)}{radius_str}")


def main():
    """Main menu."""
    while True:
        print("\n" + "="*60)
        print("BRANCH SET MANAGER")
        print("="*60)
        print("\n1. Create new branch set")
        print("2. View branch set")
        print("3. Delete branch set")
        print("4. List all branch sets")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            create_branch_set()
        elif choice == '2':
            view_branch_set()
        elif choice == '3':
            delete_branch_set()
        elif choice == '4':
            list_branch_sets()
        elif choice == '5':
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    main()
