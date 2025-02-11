import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import linalg, stats
import os

distance_threshold=0.2

def get_direction_from_points(points):
    x = points[:, 0]
    y = points[:, 1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    direction = np.array([1.0, slope])
    return direction / np.linalg.norm(direction)

def reconstruct_branch_axis(A, points_2d):
    n_cameras = len(points_2d)  # Use number of provided point sets
    A_subset = A[:, :n_cameras]  # Use corresponding subset of A matrix
    midline_vectors_2d = np.zeros((n_cameras, 2))
    
    for i in range(n_cameras):
        direction = get_direction_from_points(points_2d[i])
        midline_vectors_2d[i] = direction
    
    M = np.zeros((2 * n_cameras, 3 + n_cameras))
    
    for i in range(n_cameras):
        L = A_subset[:, i]  # Use subset of A matrix
        u, v = midline_vectors_2d[i]
        
        row = 2 * i
        M[row, 0] = (L[0] - L[3]*L[8]) / (L[8] + 1)
        M[row, 1] = (L[1] - L[3]*L[9]) / (L[9] + 1)
        M[row, 2] = (L[2] - L[3]*L[10]) / (L[10] + 1)
        M[row, 3 + i] = -u
        
        M[row + 1, 0] = (L[4] - L[7]*L[8]) / (L[8] + 1)
        M[row + 1, 1] = (L[5] - L[7]*L[9]) / (L[9] + 1)
        M[row + 1, 2] = (L[6] - L[7]*L[10]) / (L[10] + 1)
        M[row + 1, 3 + i] = -v

    U, s, Vh = linalg.svd(M, full_matrices=False)
    direction = Vh[-1, :3]
    direction = direction / np.linalg.norm(direction)
    
    centroid_2d = np.mean(points_2d[0], axis=0)
    u, v = centroid_2d
    
    P = np.zeros((2 * n_cameras, 3))
    b = np.zeros(2 * n_cameras)
    
    for i in range(n_cameras):
        L = A[:, i]
        u, v = np.mean(points_2d[i], axis=0)
        
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
    return direction, point_on_line

def reconstruct_3d_point(A, points_2d):
    n_cameras = len(points_2d)  # Use number of provided point sets
    if n_cameras < 2:
        raise ValueError("Need at least 2 cameras for 3D reconstruction")
    
    A_subset = A[:, :n_cameras]  # Use corresponding subset of A matrix
    P = np.zeros((2 * n_cameras, 3))
    b = np.zeros(2 * n_cameras)
    
    for i in range(n_cameras):
        L = A_subset[:, i]
        u, v = points_2d[i]
        
        row = 2 * i
        P[row, 0] = u*L[8] - L[0]
        P[row, 1] = u*L[9] - L[1]
        P[row, 2] = u*L[10] - L[2]
        b[row] = L[3] - u
        
        P[row + 1, 0] = v*L[8] - L[4]
        P[row + 1, 1] = v*L[9] - L[5]
        P[row + 1, 2] = v*L[10] - L[6]
        b[row + 1] = L[7] - v
    
    point_3d = np.linalg.lstsq(P, b, rcond=1e-10)[0]
    return point_3d

def create_branch_cylinder(axis_direction, axis_point, surface_points_3d, n_segments=50, n_z_segments=30):
    # Create coordinate system
    z_axis = axis_direction / np.linalg.norm(axis_direction)
    x_axis = np.array([-z_axis[1], z_axis[0], 0])
    if np.all(x_axis == 0):
        x_axis = np.array([1, 0, 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Get axis extent
    relative_points = surface_points_3d - axis_point
    axial_distances = np.dot(relative_points, z_axis)
    min_z = np.min(axial_distances)
    max_z = np.max(axial_distances)
    
    # Project points to get radii at different positions
    projected_points = relative_points - np.outer(np.dot(relative_points, z_axis), z_axis)
    radii = np.linalg.norm(projected_points, axis=1)
    
    # Create vertices
    theta = np.linspace(0, 2*np.pi, n_segments)
    z_levels = np.linspace(0, 1, n_z_segments)
    vertices = []
    
    for z_param in z_levels:
        z = min_z + z_param * (max_z - min_z)
        # Interpolate radius based on position
        weights = np.exp(-0.5 * ((z - axial_distances) / ((max_z - min_z) * 0.2))**2)
        radius = np.average(radii, weights=weights)
        
        for t in theta:
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            point = axis_point + z*z_axis + x*x_axis + y*y_axis
            vertices.append(point)
    
    vertices = np.array(vertices)
    
    # Create faces
    faces = []
    for j in range(n_z_segments - 1):
        for i in range(n_segments - 1):
            v0 = j*n_segments + i
            v1 = j*n_segments + (i + 1)
            v2 = (j + 1)*n_segments + i
            v3 = (j + 1)*n_segments + (i + 1)
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
        
        # Connect last vertices in ring
        v0 = (j + 1)*n_segments - 1
        v1 = j*n_segments
        v2 = (j + 2)*n_segments - 1
        v3 = (j + 1)*n_segments
        faces.append([v0, v1, v2])
        faces.append([v1, v3, v2])
    
    faces = np.array(faces)
    mean_radius = np.mean(radii)  # For compatibility with existing code
    
    return vertices, faces, mean_radius


def point_to_mesh_distance(point, vertices, faces):
    # Calculate minimum distance from point to mesh triangles
    min_dist = float('inf')
    for face in faces:
        triangle = vertices[face]
        # Get closest point on triangle
        dist = distance_to_triangle(point, triangle)
        min_dist = min(min_dist, dist)
    return min_dist

def distance_to_triangle(point, triangle):
    # Calculate distance from point to triangle using barycentric coordinates
    v0 = triangle[1] - triangle[0]
    v1 = triangle[2] - triangle[0]
    v2 = point - triangle[0]
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return np.inf
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    if (u >= 0) and (v >= 0) and (w >= 0):
        # Point projects inside triangle
        projected = triangle[0] + v * v0 + w * v1
        return np.linalg.norm(point - projected)
    
    # Point projects outside triangle - calculate distance to edges
    edges = [(triangle[0], triangle[1]), 
             (triangle[1], triangle[2]), 
             (triangle[2], triangle[0])]
    
    return min(distance_to_line_segment(point, edge[0], edge[1]) 
              for edge in edges)

def distance_to_line_segment(point, v1, v2):
    segment = v2 - v1
    length_sq = np.dot(segment, segment)
    if length_sq == 0:
        return np.linalg.norm(point - v1)
    
    t = max(0, min(1, np.dot(point - v1, segment) / length_sq))
    projection = v1 + t * segment
    return np.linalg.norm(point - projection)

def analyze_ant_mobility(excel_path, vertices, faces, distance_threshold):
    points_of_interest = list(range(8, 11)) + list(range(14, 17))
    mobility_data = []
    
    xl = pd.ExcelFile(excel_path)
    print("Available sheets:", xl.sheet_names)
    
    for point in points_of_interest:
        sheet_name = f"Point {point}"
        try:
            point_data = pd.read_excel(xl, sheet_name)
            print(f"Processing Point {point}, columns:", point_data.columns.tolist())
            
            # Convert column names to proper case
            column_map = {}
            for col in point_data.columns:
                col_lower = col.lower()
                if col_lower == 'time frame':
                    point_data['Time'] = point_data[col]
                    point_data['Frame'] = range(1, len(point_data) + 1)
                elif col.upper() == 'X':
                    column_map[col] = 'X'
                elif col.upper() == 'Y':
                    column_map[col] = 'Y'
                elif col.upper() == 'Z':
                    column_map[col] = 'Z'
                elif 'residual' in col_lower:
                    column_map[col] = 'Residual'
                elif 'cameras' in col_lower:
                    column_map[col] = 'Cameras Used'
            
            point_data = point_data.rename(columns=column_map)
            required_cols = ['Frame', 'Time', 'X', 'Y', 'Z']
            if not all(col in point_data.columns for col in required_cols):
                missing = [col for col in required_cols if col not in point_data.columns]
                raise ValueError(f"Missing required columns: {missing}")
                
            for _, row in point_data.iterrows():
                point_3d = np.array([row['X'], row['Y'], row['Z']])
                distance = point_to_mesh_distance(point_3d, vertices, faces)
                
                mobility_data.append({
                    'Frame': int(row['Frame']),
                    'Point': point,
                    'Time': row['Time'],
                    'X': row['X'],
                    'Y': row['Y'],
                    'Z': row['Z'],
                    'Distance_to_mesh': distance,
                    'State': 'Immobile' if distance < distance_threshold else 'Mobile',
                    'Residual': row.get('Residual', None),
                    'Cameras': row.get('Cameras', None)
                })
        except Exception as e:
            print(f"Error processing Point {point}: {str(e)}")
            continue
    
    if not mobility_data:
        raise ValueError("No valid data points found")
    
    return pd.DataFrame(mobility_data)

def main():
    # First run branch reconstruction to get vertices and faces

    # Camera coefficients
    A = np.array([
        [90.798, 91.6283, 68.6801, -6.9384],
        [0.9664, -2.8374, 5.6441, 114.8824],
        [24.4139, -17.6627, -59.2024, -29.8135],
        [594.3277, 667.1884, 676.3614, 498.9457],
        [-1.7397, 1.7846, -0.6094, -77.332],
        [149.6519, 147.3786, 143.9186, -3.9629],
        [0.121, -0.7585, 4.8412, 8.6032],
        [331.1221, 300.1799, 342.79, 590.4905],
        [0.0006, 0.001, -0.0011, -0.001],
        [-0.0012, 0.0003, 0.0006, 0.0003],
        [-0.0005, -0.0019, -0.0013, -0.0011]
    ])
    
    
    # Example points from each camera view for branch reconstruction
    points_2d = [
        np.array([
            [579.56, 1015.50],
            [579.56, 952.50],
            [578.30, 847.93],
            [577.04, 767.29],
            [574.52, 666.50],
            [573.26, 573.26],
            [572.00, 488.85],
            [569.48, 417.03],
            [568.22, 332.62],
            [565.70, 243.16],
            [565.70, 163.79],
            [564.44, 75.60],
            [563.18, 0.00]
        ]), # Left
        np.array([
            [759.73, 1015.50],
            [758.47, 923.52],
            [755.95, 817.69],
            [754.69, 687.92],
            [752.17, 560.66],
            [750.91, 432.15],
            [749.65, 326.32],
            [748.39, 229.31],
            [745.87, 118.43],
            [744.61, -1.26]
        ]), # Top
        np.array([
            [882.59, 761.95],
            [873.07, 641.31],
            [866.72, 527.02],
            [861.96, 436.54],
            [857.20, 355.58],
            [852.43, 276.21],
            [846.08, 192.08],
            [842.91, 112.71],
            [834.97, 3.17]
        ]), # Right
        #np.array([[574.64, 961.96], [507.97, 633.37], [541.30, 803.22], [439.71, 280.97]]) # Front
    ]
    
    # Surface points for branch reconstruction (L, T, R, F)
    surface_points_2d = [
        [[798.70, 964.67], [798.79, 932.34], [749.05, 974.48], [933.57, 446.15]],
        [[614.84, 546.81], [622.40, 514.05], [587.12, 573.26], [617.53, 613.34]],
        [[723.44, 503.85], [732.07, 476.20], [669.16, 531.72], [577.04, 522.87]],
        [[700.52, 113.39], [718.15, 94.49], [635.41, 164.30], [280.50, 552.00]]
    ]

    print("Reconstructing branch...")
    axis_direction, axis_point = reconstruct_branch_axis(A, points_2d)
    surface_points_3d = np.array([reconstruct_3d_point(A, point_2d) for point_2d in surface_points_2d])
    vertices, faces, radius = create_branch_cylinder(axis_direction, axis_point, surface_points_3d)
    
    print("Analyzing mobility...")
    mobility_df = analyze_ant_mobility(
        '/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data/11U1.xlsx',
        vertices, 
        faces,
        distance_threshold
    )
    
    # Save to Excel with multiple sheets
    output_path = '/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data/branch_mobility/ant_mobility_analysis.xlsx'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with pd.ExcelWriter(output_path) as writer:
        # Overall summary
        mobility_df.to_excel(writer, sheet_name='All_Points', index=False)
        
        # Individual point summaries
        for point in mobility_df['Point'].unique():
            point_data = mobility_df[mobility_df['Point'] == point]
            point_data.to_excel(writer, sheet_name=f'Point_{point}', index=False)
        
        # Summary statistics
        summary_stats = pd.DataFrame([{
            'Point': point,
            'Total_Frames': len(point_data),
            'Immobile_Frames': len(point_data[point_data['State'] == 'Immobile']),
            'Mobile_Frames': len(point_data[point_data['State'] == 'Mobile']),
            'Percent_Immobile': (len(point_data[point_data['State'] == 'Immobile']) / len(point_data) * 100),
            'Mean_Distance': point_data['Distance_to_mesh'].mean(),
            'Max_Distance': point_data['Distance_to_mesh'].max()
        } for point, point_data in mobility_df.groupby('Point')])
        
        summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)

if __name__ == "__main__":
    main()