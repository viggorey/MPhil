import os
import pandas as pd
import numpy as np

distance_threshold = 0.15

# Path to the directory containing the Excel files
directory = "/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data"

# Output directory
output_directory = os.path.join(directory, "metadata_combined")
os.makedirs(output_directory, exist_ok=True)

def calculate_metadata(file_path):
    try:
        # Specify the engine explicitly
        excel_data = pd.ExcelFile(file_path, engine='openpyxl')
        
        # Load sheets for all points
        all_points = {f"Point {i}": pd.read_excel(excel_data, sheet_name=f"Point {i}") for i in range(1, 17)}
        
        # Extract specific points for analysis
        data_point_1 = all_points["Point 1"]
        data_point_2 = all_points["Point 2"]
        data_point_3 = all_points["Point 3"]
        
        foot_tracks = ["Point 8", "Point 9", "Point 10", "Point 14", "Point 15", "Point 16"]
        foot_data = {track: all_points[track] for track in foot_tracks}
        
        # Calculate speed along Y-axis for Point 1
        data_point_1['Y Speed'] = data_point_1['Y'].diff() / data_point_1['Time Frame'].diff()
        
        # Calculate midpoint coordinates between Points 2 and 3
        midpoint_x = (data_point_2['X'] + data_point_3['X']) / 2
        midpoint_y = (data_point_2['Y'] + data_point_3['Y']) / 2
        midpoint_z = (data_point_2['Z'] + data_point_3['Z']) / 2
        midpoint_combined = midpoint_x.map("{:.3f}".format) + ", " + \
                            midpoint_y.map("{:.3f}".format) + ", " + \
                            midpoint_z.map("{:.3f}".format)
        
        # Calculate distances, vector movement, and movement states for each foot
        distances = {}
        vector_movements = {}
        movement_states = {}
        branch_coordinates = pd.DataFrame(columns=['X', 'Y', 'Z', 'Time Frame', 'Track'])
        
        for track, track_data in foot_data.items():
            # Distance to the midpoint
            distances[track] = np.sqrt(
                (track_data['X'] - midpoint_x) ** 2 +
                (track_data['Y'] - midpoint_y) ** 2 +
                (track_data['Z'] - midpoint_z) ** 2
            )
            
            # Vector movement distance
            vector_movement = np.sqrt(
                track_data['X'].diff() ** 2 +
                track_data['Y'].diff() ** 2 +
                track_data['Z'].diff() ** 2
            )
            vector_movements[f"{track} Movement"] = vector_movement
            
            # Movement states
            movement_state = []
            immobile_counter = 0
            for idx, dist in enumerate(vector_movement):
                if dist <= distance_threshold:
                    immobile_counter += 1
                else:
                    immobile_counter = 0
                
                if immobile_counter >= 2:
                    movement_state.append("Immobile")
                    # Add immobile coordinates to branch dataset
                    new_entry = pd.DataFrame({
                        'X': [track_data['X'].iloc[idx]],
                        'Y': [track_data['Y'].iloc[idx]],
                        'Z': [track_data['Z'].iloc[idx]],
                        'Time Frame': [track_data['Time Frame'].iloc[idx]],
                        'Track': [track]
                    })

                    # Only concatenate if the new entry is not empty
                    if not new_entry.isna().all().all():
                        branch_coordinates = pd.concat([branch_coordinates, new_entry], ignore_index=True)
                else:
                    movement_state.append("Mobile")
            
            movement_states[f"{track} State"] = movement_state
        
        # Combine metadata into a single DataFrame
        combined_data = pd.DataFrame()
        combined_data['Time Frame'] = data_point_1['Time Frame']
        combined_data['Y Speed'] = data_point_1['Y Speed']
        combined_data['Midpoint (X, Y, Z)'] = midpoint_combined
        
        for track, distance in distances.items():
            combined_data[f"{track} Distance"] = distance
        
        for track, movement in vector_movements.items():
            combined_data[track] = movement
        
        for state_name, state_data in movement_states.items():
            combined_data[state_name] = state_data
        
        # Create a separate sheet for all 3D coordinates of Points 1–16
        all_3d_data = pd.DataFrame()
        all_3d_data['Time Frame'] = data_point_1['Time Frame']
        for point, point_data in all_points.items():
            all_3d_data[f"{point} (X, Y, Z)"] = point_data['X'].map("{:.3f}".format) + ", " + \
                                                point_data['Y'].map("{:.3f}".format) + ", " + \
                                                point_data['Z'].map("{:.3f}".format)
        
        # Save the data to an Excel file
        file_name = file_path.split("/")[-1].replace(".xlsx", "_Combined_Metadata.xlsx")
        output_path = os.path.join(output_directory, file_name)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            combined_data.to_excel(writer, sheet_name='Metadata', index=False)
            all_3d_data.to_excel(writer, sheet_name='All 3D Coordinates', index=False)
            branch_coordinates.to_excel(writer, sheet_name='Branch', index=False)
        
        print(f"Metadata saved to {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Iterate through all Excel files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(directory, filename)
        calculate_metadata(file_path)