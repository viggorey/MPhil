"""
DLT (Direct Linear Transformation) utility functions for 3D reconstruction.
Based on A1_DLTCalibration.m and B2_3DAnts2.m
"""

import numpy as np
from typing import Tuple, List, Optional
from itertools import combinations


def calculate_dlt_coefficients(F: np.ndarray, L: np.ndarray, 
                               cut_points: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
    """
    Calculate DLT coefficients for one camera.
    
    Parameters:
    -----------
    F : np.ndarray
        Matrix containing the global coordinates (X, Y, Z) of calibration points
        Shape: (n_points, 3)
    L : np.ndarray
        Matrix containing 2D coordinates of calibration points seen in camera
        Shape: (n_points, 2) - columns are [u, v] image coordinates
    cut_points : List[int], optional
        Indices of points to exclude (1-indexed, will be converted to 0-indexed)
    
    Returns:
    --------
    A : np.ndarray
        11 DLT coefficients
    avg_residual : float
        Average residual (measure for fit of DLT) in units of camera coordinates
    """
    if cut_points is None:
        cut_points = []
    
    if F.shape[0] != L.shape[0]:
        raise ValueError("Number of calibration points in F and L do not agree")
    
    if F.shape[0] < 6:
        raise ValueError("Need at least 6 calibration points for DLT calculation")
    
    m = F.shape[0]
    
    # Flatten L into column vector: [u1, v1, u2, v2, ...]
    # MATLAB's L(:) flattens column-wise (Fortran order), so we need 'F' flag
    L_flat = L.T.flatten('F')  # Fortran-style (column-major) flattening to match MATLAB
    
    # Build the B matrix
    B = np.zeros((2 * m, 11))
    
    for i in range(m):
        x, y, z = F[i, 0], F[i, 1], F[i, 2]
        u, v = L[i, 0], L[i, 1]
        
        # Row for u coordinate
        B[2*i, 0] = x
        B[2*i, 1] = y
        B[2*i, 2] = z
        B[2*i, 3] = 1
        B[2*i, 8] = -x * u
        B[2*i, 9] = -y * u
        B[2*i, 10] = -z * u
        
        # Row for v coordinate
        B[2*i + 1, 4] = x
        B[2*i + 1, 5] = y
        B[2*i + 1, 6] = z
        B[2*i + 1, 7] = 1
        B[2*i + 1, 8] = -x * v
        B[2*i + 1, 9] = -y * v
        B[2*i + 1, 10] = -z * v
    
    # Remove cut points (convert from 1-indexed to 0-indexed)
    if cut_points:
        cut_indices = []
        for cp in cut_points:
            idx = cp - 1  # Convert to 0-indexed
            if 0 <= idx < m:
                cut_indices.extend([2*idx, 2*idx + 1])
        
        B = np.delete(B, cut_indices, axis=0)
        L_flat = np.delete(L_flat, cut_indices)
    
    # Solve for DLT coefficients
    A = np.linalg.lstsq(B, L_flat, rcond=1e-10)[0]
    
    # Back-calculate the u,v image coordinates
    D = B @ A
    
    # Calculate residuals
    R = L_flat - D
    residual_norm = np.linalg.norm(R)
    avg_residual = residual_norm / np.sqrt(len(R))
    
    # Correct for mirroring (flip first coefficient)
    A[0] = -A[0]
    
    return A, avg_residual


def reconstruct_3d_point(A: np.ndarray, points_2d: List[np.ndarray]) -> np.ndarray:
    """
    Reconstruct a 3D point from 2D points in multiple camera views.
    
    Parameters:
    -----------
    A : np.ndarray
        Camera coefficients matrix, shape (11, n_cameras)
    points_2d : List[np.ndarray]
        List of 2D points [u, v] for each camera (can be empty arrays for missing cameras)
    
    Returns:
    --------
    point_3d : np.ndarray
        3D coordinates [X, Y, Z]
    """
    # Filter out empty arrays and get valid camera indices
    valid_cameras = [i for i, points in enumerate(points_2d) if points.size > 0]
    
    if len(valid_cameras) < 2:
        raise ValueError("Need at least 2 cameras with points for reconstruction")
    
    # Use only valid cameras
    A_subset = A[:, valid_cameras]
    n_cameras = len(valid_cameras)
    
    # Build the system of equations
    P = np.zeros((2 * n_cameras, 3))
    b = np.zeros(2 * n_cameras)
    
    for i, cam_idx in enumerate(valid_cameras):
        L = A_subset[:, i]
        u, v = points_2d[cam_idx]
        
        row = 2 * i
        P[row, 0] = u * L[8] - L[0]
        P[row, 1] = u * L[9] - L[1]
        P[row, 2] = u * L[10] - L[2]
        b[row] = L[3] - u
        
        P[row + 1, 0] = v * L[8] - L[4]
        P[row + 1, 1] = v * L[9] - L[5]
        P[row + 1, 2] = v * L[10] - L[6]
        b[row + 1] = L[7] - v
    
    # Solve for 3D point
    point_3d = np.linalg.lstsq(P, b, rcond=1e-10)[0]
    
    return point_3d


def calculate_geometric_score(camera_indices: List[int]) -> float:
    """
    Calculate geometric score for a camera combination.
    Based on B2_3DAnts2.m calculateGeometricScore function.
    
    Parameters:
    -----------
    camera_indices : List[int]
        List of camera indices (0=Left, 1=Top, 2=Right, 3=Front)
    
    Returns:
    --------
    score : float
        Geometric score (higher is better, closer to 90 degrees between cameras is better)
    """
    # Define approximate camera positions (order: L, T, R, F)
    camera_positions = [
        np.array([-0.5, 0.866, 0]),    # Left (30 degrees to left)
        np.array([0, 1, 0]),            # Top (directly above)
        np.array([0.5, 0.866, 0]),    # Right (30 degrees to right)
        np.array([0, 0.940, 0.342])    # Front (20 degrees above)
    ]
    
    # Calculate angles between cameras
    angles = []
    for i in range(len(camera_indices)):
        for j in range(i + 1, len(camera_indices)):
            v1 = camera_positions[camera_indices[i]]
            v2 = camera_positions[camera_indices[j]]
            # Calculate angle between camera vectors (in degrees)
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)  # Clamp to valid range
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)
    
    # Score based on angular distribution (closer to 90 degrees is better)
    if len(angles) == 0:
        return 0.0
    score = np.mean(1 - np.abs(np.array(angles) - 90) / 90)
    return score


def reconstruct_3d_point_with_residual(A: np.ndarray, points_2d_list: List[np.ndarray], 
                                       camera_indices: List[int]) -> Tuple[np.ndarray, float]:
    """
    Reconstruct a 3D point from 2D points and calculate DOF-based residual.
    Based on B2_3DAnts2.m reconstruction logic.
    
    Parameters:
    -----------
    A : np.ndarray
        Camera coefficients matrix, shape (11, n_cameras)
    points_2d_list : List[np.ndarray]
        List of 2D points [u, v] for each camera in the combination (same order as camera_indices)
    camera_indices : List[int]
        List of camera indices used in this reconstruction
    
    Returns:
    --------
    point_3d : np.ndarray
        3D coordinates [X, Y, Z]
    residual : float
        DOF-based residual
    """
    n_cameras = len(camera_indices)
    
    if len(points_2d_list) != n_cameras:
        raise ValueError(f"Number of 2D points ({len(points_2d_list)}) must match number of cameras ({n_cameras})")
    
    # Build the system of equations (matching B2_3DAnts2.m format)
    L1 = np.zeros((2 * n_cameras, 3))
    L2 = np.zeros(2 * n_cameras)
    
    for i, cam_idx in enumerate(camera_indices):
        L = A[:, cam_idx]
        point_2d = points_2d_list[i]
        # Ensure point_2d is a numpy array and extract u, v
        if isinstance(point_2d, np.ndarray):
            if point_2d.size == 0:
                raise ValueError(f"Empty 2D point for camera {cam_idx}")
            u, v = float(point_2d[0]), float(point_2d[1])
        else:
            raise ValueError(f"Invalid point type for camera {cam_idx}: {type(point_2d)}")
        
        # Row for u coordinate
        L1[2 * i, 0] = L[0] - u * L[8]
        L1[2 * i, 1] = L[1] - u * L[9]
        L1[2 * i, 2] = L[2] - u * L[10]
        L2[2 * i] = u - L[3]
        
        # Row for v coordinate
        L1[2 * i + 1, 0] = L[4] - v * L[8]
        L1[2 * i + 1, 1] = L[5] - v * L[9]
        L1[2 * i + 1, 2] = L[6] - v * L[10]
        L2[2 * i + 1] = v - L[7]
    
    # Solve for 3D point
    point_3d = np.linalg.lstsq(L1, L2, rcond=1e-10)[0]
    
    # Calculate DOF-based residual (matching B2_3DAnts2.m)
    h = L1 @ point_3d
    DOF = (n_cameras * 2 - 3)  # Degrees of freedom
    residual = np.sqrt(np.sum((L2 - h) ** 2) / DOF)
    
    return point_3d, residual


def reconstruct_3d_point_best_combination(A: np.ndarray, points_2d: List[np.ndarray],
                                          valid_camera_indices: List[int],
                                          camera_labels: List[str]) -> Tuple[np.ndarray, float, str]:
    """
    Reconstruct 3D point using best camera combination (B2_3DAnts2.m logic).
    Tests all possible combinations, scores them, and selects the best.
    
    Parameters:
    -----------
    A : np.ndarray
        Camera coefficients matrix, shape (11, n_cameras)
    points_2d : List[np.ndarray]
        List of 2D points [u, v] for each camera (can be empty for missing cameras)
    valid_camera_indices : List[int]
        List of valid camera indices (0-indexed)
    camera_labels : List[str]
        Camera labels (e.g., ['L', 'T', 'R', 'F'])
    
    Returns:
    --------
    point_3d : np.ndarray
        Best 3D coordinates [X, Y, Z]
    residual : float
        Residual for best combination
    combo_name : str
        Name of best camera combination (e.g., 'LTRF')
    """
    if len(valid_camera_indices) < 2:
        raise ValueError("Need at least 2 cameras with points for reconstruction")
    
    all_results = []
    
    # Generate all possible camera combinations (2, 3, 4, ... cameras)
    for num_cams in range(2, len(valid_camera_indices) + 1):
        for combo in combinations(valid_camera_indices, num_cams):
            combo = list(combo)
            
            # Extract 2D points for this combination
            combo_points_2d = [points_2d[i] for i in combo]
            
            # Reconstruct 3D point and calculate residual
            try:
                point_3d, residual = reconstruct_3d_point_with_residual(
                    A, combo_points_2d, combo
                )
                
                # Calculate geometric score
                geometric_score = calculate_geometric_score(combo)
                
                # Create combo name
                combo_name = ''.join([camera_labels[i] for i in combo])
                
                all_results.append({
                    'combo': combo_name,
                    'point_3d': point_3d,
                    'residual': residual,
                    'num_cams': len(combo),
                    'geometric_score': geometric_score
                })
            except Exception:
                continue
    
    if not all_results:
        raise ValueError("No valid camera combinations found")
    
    # Extract arrays for scoring
    residuals = np.array([r['residual'] for r in all_results])
    num_cams = np.array([r['num_cams'] for r in all_results])
    geometric_scores = np.array([r['geometric_score'] for r in all_results])
    
    # Normalize scores
    norm_residuals = residuals / (np.max(residuals) + 1e-10)
    norm_cams = num_cams / (np.max(num_cams) + 1e-10)
    norm_geometric = geometric_scores / (np.max(geometric_scores) + 1e-10)
    
    # Combined score (weighted sum, matching B2_3DAnts2.m line 182)
    # Higher score is better: lower residual (negative), more cameras, better geometry
    combined_scores = -0.2 * norm_residuals + 0.5 * norm_cams + 0.3 * norm_geometric
    
    # Select best combination
    best_idx = np.argmax(combined_scores)
    best_result = all_results[best_idx]
    
    return best_result['point_3d'], best_result['residual'], best_result['combo']
