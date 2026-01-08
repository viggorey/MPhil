"""
DLT (Direct Linear Transformation) utility functions for 3D reconstruction.
Based on A1_DLTCalibration.m
"""

import numpy as np
from typing import Tuple, List, Optional


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
    L_flat = L.T.flatten()
    
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
