import plotly.graph_objects as go
import pandas as pd
import numpy as np
import openpyxl

def create_legend_trace(color, name):
    return go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color=color),
        name=name,
        showlegend=True
    )

def read_excel_data():
    metadata_path = '/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data/metadata_combined/11U1_Combined_Metadata.xlsx'
    mobility_path = '/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data/branch_mobility/ant_mobility_analysis.xlsx'
    coords_df = pd.read_excel(metadata_path, sheet_name='All 3D Coordinates')
    states_df1 = pd.read_excel(metadata_path, sheet_name='Metadata')
    states_df2 = pd.read_excel(mobility_path, sheet_name='All_Points')
    return coords_df, states_df1, states_df2

def parse_coords(coord_str):
    return [float(x.strip()) for x in coord_str.split(',')]

def get_state_info(point, frame, states_df1, states_df2):
    state1 = states_df1.loc[states_df1['Time Frame'] == frame, f'Point {point} State'].iloc[0]
    state2 = states_df2.loc[(states_df2['Frame'] == frame) & (states_df2['Point'] == point), 'State'].iloc[0]
    return state1, state2

def get_color(state1, state2):
    if state1 == state2:
        return '#006400' if state1 == 'Mobile' else '#90EE90'
    else:
        return '#ffcccb' if state1 == 'Mobile' else '#8b0000'

def create_3d_plot(coords_df, states_df1, states_df2):
    fig = go.Figure()
    
    # Add legend traces first
    fig.add_trace(create_legend_trace('#006400', 'Matching Mobile'))
    fig.add_trace(create_legend_trace('#90EE90', 'Matching Immobile'))
    fig.add_trace(create_legend_trace('#ffcccb', 'Mismatch (Feet mov Mobile)'))
    fig.add_trace(create_legend_trace('#8b0000', 'Mismatch (Feet mov Immobile)'))
    fig.add_trace(create_legend_trace('gray', 'Other Points'))
    
    state_points = [8, 9, 10, 14, 15, 16]
    
    for point in range(1, 17):
        col_name = f'Point {point} (X, Y, Z)'
        coords = [parse_coords(coord) for coord in coords_df[col_name]]
        x, y, z = zip(*coords)
        
        if point in state_points:
            # Add connecting lines
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(
                    color='rgba(100, 100, 100, 0.6)',
                    width=2
                ),
                showlegend=False
            ))
            
            # Add points
            colors = []
            for frame in range(1, len(coords_df) + 1):
                try:
                    state1, state2 = get_state_info(point, frame, states_df1, states_df2)
                    colors.append(get_color(state1, state2))
                except:
                    colors.append('#ffff00')
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors,
                    opacity=0.8
                ),
                name=f'Point {point}',
                showlegend=True
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=3,
                    color='gray',
                    opacity=0.3
                ),
                name=f'Point {point}',
                showlegend=True
            ))
    
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        title="3D Trajectory with State Comparison",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    fig.show()

if __name__ == "__main__":
    coords_df, states_df1, states_df2 = read_excel_data()
    create_3d_plot(coords_df, states_df1, states_df2)