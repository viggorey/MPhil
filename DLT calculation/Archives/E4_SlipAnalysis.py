import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy import stats

# ===== CONFIGURATION =====
# Path to folder containing metadata files
DATA_FOLDER = "/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/3D transformation/5. Datasets/3D data/Large branch"

# Groups to compare (separated by semicolons, each group can be multiple conditions separated by commas)
# Examples:
#   "11U,21U" - Compare wax runners vs non-wax runners on waxy stems going up
#   "11U,21U;11D,21D" - Compare both up and down movements
#   "11U,12U" - Compare wax runners on waxy vs non-waxy stems going up
COMPARISON_GROUPS = "11U,12U,21U,22U"

# Number of frames to analyze after slip event
FRAMES_AFTER_SLIP = 5

# Available metrics for analysis (uncomment the ones you want to analyze)
METRICS = {
    'Kinematics': [
        # 'Speed (mm/s)',
        'Point_1_branch_distance',
        'Point_2_branch_distance',
        'Point_3_branch_distance',
        'Point_4_branch_distance'
    ],
    'Behavioral_Scores': [
        # 'Gaster_Dorsal_Ventral_Angle',
        # 'Gaster_Left_Right_Angle',
        # 'Head_Distance_Foot_8',
        # 'Head_Distance_Foot_14'
    ],
    'CoM': [
        # 'CoM_Y',  # Vertical movement
        # 'CoM_X',  # Forward/backward movement
        # 'CoM_Z'   # Left/right movement
    ],
    'Duty_Factor': [
        # 'foot_8_attached',
        # 'foot_9_attached',
        # 'foot_10_attached',
        # 'foot_14_attached',
        # 'foot_15_attached',
        # 'foot_16_attached'
    ]
}

# Color scheme for groups
COLOR_SCHEME = {
    '11U': '#1f77b4',  # blue
    '21U': '#ff7f0e',  # orange
    '12U': '#2ca02c',  # green
    '22U': '#d62728',  # red
    '11D': '#9467bd',  # purple
    '21D': '#8c564b',  # brown
    '12D': '#e377c2',  # pink
    '22D': '#7f7f7f'   # gray
}

# Base output folder structure
OUTPUT_FOLDER = "comparison_plots"
SLIP_ANALYSIS_FOLDER = "Slip_Analysis"
BEHAVIORAL_SCORES_FOLDER = "Behavioral_Scores"

def sanitize_filename(filename):
    """Replace problematic characters in filenames with underscores."""
    problematic_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in problematic_chars:
        filename = filename.replace(char, '_')
    return filename

def load_metadata_file(file_path):
    """Load all sheets from a metadata Excel file."""
    try:
        return pd.read_excel(file_path, sheet_name=None)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def parse_group_name(filename):
    """Extract group information from filename (e.g., meta_11U2.xlsx -> 11U)."""
    base_name = os.path.basename(filename)
    if base_name.startswith('meta_'):
        return base_name[5:8]  # Extract the group code (e.g., 11U)
    return None

def find_slip_events(data, sheet_name='Behavioral_Scores', column_name='Slip_Score', threshold=2):
    """Find frames where slip score is above threshold."""
    slip_scores = data[sheet_name][column_name]
    slip_events = []
    
    for i in range(len(slip_scores)):
        if slip_scores[i] >= threshold:
            # Get the frame and the following frames
            frames = list(range(i, min(i + FRAMES_AFTER_SLIP, len(slip_scores))))
            slip_events.append(frames)
    
    return slip_events

def analyze_slip_events(data, slip_events, metrics):
    """Analyze metrics during and after slip events."""
    results = {}
    
    for sheet_name, metric_list in metrics.items():
        results[sheet_name] = {}
        for metric in metric_list:
            if metric in data[sheet_name].columns:
                metric_data = data[sheet_name][metric]
                event_data = []
                
                for event_frames in slip_events:
                    event_values = [metric_data[frame] for frame in event_frames]
                    event_data.append(event_values)
                
                results[sheet_name][metric] = event_data
    
    return results

def create_slip_event_plots(data_dict, metric_name, output_path):
    """Create plots showing metric values during and after slip events."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])
    
    # Plot individual slip events
    for group, group_data in data_dict.items():
        color = COLOR_SCHEME.get(group, '#000000')
        for event_idx, event_data in enumerate(group_data):
            # Only add label for the first event of each group to avoid crowded legend
            if event_idx == 0:
                ax1.plot(range(len(event_data)), event_data, 
                        color=color, alpha=0.3, 
                        label=group)
            else:
                ax1.plot(range(len(event_data)), event_data, 
                        color=color, alpha=0.3)
    
    # Plot mean with SD
    for group, group_data in data_dict.items():
        color = COLOR_SCHEME.get(group, '#000000')
        # Convert all events to same length (pad with NaN)
        max_length = max(len(event) for event in group_data)
        padded_data = np.array([np.pad(event, (0, max_length - len(event)), 
                                     constant_values=np.nan) 
                              for event in group_data])
        
        mean_data = np.nanmean(padded_data, axis=0)
        std_data = np.nanstd(padded_data, axis=0)
        
        x = np.arange(len(mean_data))
        ax2.plot(x, mean_data, color=color, label=group, linewidth=2)
        ax2.fill_between(x, mean_data - std_data, mean_data + std_data, 
                        color=color, alpha=0.2)
    
    # Customize plots
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Frames after slip event')
        ax.set_ylabel(metric_name)
    
    ax1.set_title('Individual Slip Events')
    ax2.set_title('Mean ± Standard Deviation')
    
    # Add legend to both plots
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create base output folder structure
    base_output_dir = os.path.join(script_dir, OUTPUT_FOLDER)
    slip_analysis_dir = os.path.join(base_output_dir, SLIP_ANALYSIS_FOLDER)
    behavioral_scores_dir = os.path.join(base_output_dir, BEHAVIORAL_SCORES_FOLDER)
    
    os.makedirs(slip_analysis_dir, exist_ok=True)
    os.makedirs(behavioral_scores_dir, exist_ok=True)
    
    # Find all metadata files
    metadata_files = [f for f in os.listdir(DATA_FOLDER) if f.startswith('meta_') and f.endswith('.xlsx')]
    
    # Parse comparison groups
    comparison_groups = [group.split(',') for group in COMPARISON_GROUPS.split(';')]
    
    # Load all metadata files
    metadata_dict = {}
    for file in metadata_files:
        group = parse_group_name(file)
        if group:
            file_path = os.path.join(DATA_FOLDER, file)
            metadata_dict[group] = load_metadata_file(file_path)
    
    # Process each comparison
    for comparison in comparison_groups:
        # Collect data for each group
        group_data = {}
        for group in comparison:
            if group in metadata_dict:
                group_data[group] = []
                for file in metadata_files:
                    if file.startswith(f'meta_{group}'):
                        file_path = os.path.join(DATA_FOLDER, file)
                        metadata = load_metadata_file(file_path)
                        if metadata:
                            # Find slip events
                            slip_events = find_slip_events(metadata)
                            if slip_events:
                                # Analyze metrics during slip events
                                event_data = analyze_slip_events(metadata, slip_events, METRICS)
                                group_data[group].append(event_data)
        
        # Create plots for each metric
        for sheet_name, metrics in METRICS.items():
            for metric in metrics:
                # Collect data for this metric
                metric_data = {}
                for group, group_events in group_data.items():
                    metric_data[group] = []
                    for event_data in group_events:
                        if sheet_name in event_data and metric in event_data[sheet_name]:
                            metric_data[group].extend(event_data[sheet_name][metric])
                
                if metric_data:
                    # Create plot
                    plot_name = f"{'_'.join(comparison)}_{sheet_name}_{metric}.png"
                    
                    # Determine output directory based on sheet name
                    if sheet_name == 'Behavioral_Scores':
                        output_dir = behavioral_scores_dir
                    else:
                        output_dir = slip_analysis_dir
                    
                    output_path = os.path.join(output_dir, plot_name)
                    create_slip_event_plots(metric_data, metric, output_path)
                    print(f"Created plot: {output_path}")

if __name__ == "__main__":
    main() 