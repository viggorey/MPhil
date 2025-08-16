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

# Sheet and column to compare (format: "SheetName:ColumnName")
# Examples:
#   "Duty_Factor:foot_8_attached" - Individual foot attachment
#   "Duty_Factor:foot_9_attached" - Individual foot attachment
#   "Duty_Factor:foot_10_attached" - Individual foot attachment
#   "Duty_Factor:foot_14_attached" - Individual foot attachment
#   "Duty_Factor:foot_15_attached" - Individual foot attachment
#   "Duty_Factor:foot_16_attached" - Individual foot attachment
METRIC = "Duty_Factor:total_attached_legs"

# Color scheme for groups (can be modified)
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

# Base output folder (relative to script location)
OUTPUT_FOLDER = "comparison_plots"

def sanitize_filename(filename):
    """Replace problematic characters in filenames with underscores."""
    problematic_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in problematic_chars:
        filename = filename.replace(char, '_')
    return filename

def calculate_total_attached_legs(metadata, sheet_name):
    """Calculate total attached legs from individual foot attachment columns."""
    if sheet_name not in metadata:
        return None
    
    sheet = metadata[sheet_name]
    foot_columns = [col for col in sheet.columns if col.startswith('foot_') and col.endswith('_attached')]
    
    if not foot_columns:
        return None
    
    # Sum all foot attachment columns
    total_attached = sheet[foot_columns].sum(axis=1)
    return total_attached

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

def create_comparison_plot(data_dict, metric_name, output_path):
    """Create a figure with three subplots: individual lines, mean with SD, and boxplot with significance."""
    # Check if we have any data
    if not data_dict:
        print("Warning: No data provided to create_comparison_plot")
        return
    
    # Check if any group has data
    groups_with_data = {group: data for group, data in data_dict.items() if data}
    if not groups_with_data:
        print("Warning: No groups have data in create_comparison_plot")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), height_ratios=[1, 1, 1])
    
    # Plot individual lines
    for group, group_data in groups_with_data.items():
        color = COLOR_SCHEME.get(group, '#000000')  # Default to black if color not specified
        for file_name, data in group_data.items():
            ax1.plot(range(len(data)), data, color=color, alpha=0.3)
        # Add only one legend entry per group
        ax1.plot([], [], color=color, label=group, linewidth=2)
    
    # Plot mean with SD
    for group, group_data in groups_with_data.items():
        color = COLOR_SCHEME.get(group, '#000000')
        # Convert all series to same length (pad with NaN)
        max_length = max(len(data) for data in group_data.values())
        # Convert data to float type before padding
        padded_data = np.array([np.pad(data.astype(float), (0, max_length - len(data)), 
                                     constant_values=np.nan) 
                              for data in group_data.values()])
        
        mean_data = np.nanmean(padded_data, axis=0)
        std_data = np.nanstd(padded_data, axis=0)
        
        x = np.arange(len(mean_data))
        ax2.plot(x, mean_data, color=color, label=group, linewidth=2)
        ax2.fill_between(x, mean_data - std_data, mean_data + std_data, 
                        color=color, alpha=0.2)
    
    # Customize plots
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Frames')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
    
    ax1.set_title('Individual Traces')
    ax2.set_title('Mean ± Standard Deviation')
    
    # Add legend to both plots
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # --- Boxplot subplot ---
    # Gather per-individual means for each group
    boxplot_data = []
    boxplot_groups = []
    for group, group_data in groups_with_data.items():
        for file_name, data in group_data.items():
            mean_val = np.nanmean(data.astype(float))  # Convert to float here as well
            boxplot_data.append(mean_val)
            boxplot_groups.append(group)
    boxplot_df = pd.DataFrame({
        'Value': boxplot_data,
        'Group': boxplot_groups
    })
    boxplot_df = boxplot_df.dropna()

    sns.boxplot(x='Group', y='Value', data=boxplot_df, ax=ax3)
    ax3.set_title('Per-Individual Mean Values by Group (Boxplot)')
    ax3.set_ylabel(metric_name.replace('_', ' ').title())
    ax3.set_xlabel('Group')

    # --- Statistical test and annotation ---
    unique_groups = sorted(boxplot_df['Group'].unique())
    group_values = [boxplot_df[boxplot_df['Group'] == g]['Value'].values for g in unique_groups]
    if len(unique_groups) == 2:
        # Independent t-test
        stat, pval = stats.ttest_ind(group_values[0], group_values[1], nan_policy='omit')
        test_name = 't-test'
    elif len(unique_groups) > 2:
        # One-way ANOVA
        stat, pval = stats.f_oneway(*group_values)
        test_name = 'ANOVA'
    else:
        stat, pval = None, None
        test_name = ''

    # Annotate p-value
    if pval is not None:
        signif = '*' if pval < 0.05 else ''
        pval_text = f"p = {pval:.3g}{signif} ({test_name})"
        # Place annotation above the rightmost box
        y_max = boxplot_df['Value'].max()
        x_pos = len(unique_groups) - 0.05  # Just inside the last box
        ax3.text(x_pos, y_max + 0.05 * abs(y_max), pval_text, ha='right', va='bottom', fontsize=12)

    ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parse sheet and column name
    sheet_name, column_name = METRIC.split(':')
    
    # Create output folder structure
    metric_folder = os.path.join(script_dir, OUTPUT_FOLDER, sheet_name)
    os.makedirs(metric_folder, exist_ok=True)
    
    # Find all metadata files
    metadata_files = [f for f in os.listdir(DATA_FOLDER) if f.startswith('meta_') and f.endswith('.xlsx')]
    
    # Parse comparison groups - handle both single group and multiple groups
    if ';' in COMPARISON_GROUPS:
        # Multiple comparisons separated by semicolons
        comparison_groups = [group.split(',') for group in COMPARISON_GROUPS.split(';')]
    else:
        # Single comparison with multiple groups
        comparison_groups = [COMPARISON_GROUPS.split(',')]
    
    # Load all metadata files
    metadata_dict = {}
    for file in metadata_files:
        group = parse_group_name(file)
        if group:
            file_path = os.path.join(DATA_FOLDER, file)
            metadata_dict[group] = load_metadata_file(file_path)
    
    print(f"Found metadata files for groups: {list(metadata_dict.keys())}")
    
    # Process each comparison
    for comparison in comparison_groups:
        print(f"Processing comparison: {comparison}")
        
        # Collect data for the metric
        data_dict = {}
        for group in comparison:
            data_dict[group] = {}
            for file in metadata_files:
                if file.startswith(f'meta_{group}'):
                    file_path = os.path.join(DATA_FOLDER, file)
                    metadata = load_metadata_file(file_path)
                    if metadata and sheet_name in metadata:
                        if column_name == 'total_attached_legs':
                            # Calculate total attached legs
                            total_data = calculate_total_attached_legs(metadata, sheet_name)
                            if total_data is not None:
                                data_dict[group][file] = total_data
                                print(f"  Found data for {group} in {file} (calculated total)")
                        elif column_name in metadata[sheet_name].columns:
                            data_dict[group][file] = metadata[sheet_name][column_name]
                            print(f"  Found data for {group} in {file}")
        
        # Check if we have any data
        non_empty_groups = {group: data for group, data in data_dict.items() if data}
        
        if non_empty_groups:
            print(f"  Creating plot with data for groups: {list(non_empty_groups.keys())}")
            # Create plot
            plot_name = f"{'_'.join(comparison)}_{column_name}.png"
            output_path = os.path.join(metric_folder, plot_name)
            create_comparison_plot(non_empty_groups, column_name, output_path)
            print(f"Created plot: {output_path}")
        else:
            print(f"  No data found for any groups in comparison: {comparison}")
            print(f"  Available groups: {list(metadata_dict.keys())}")
            print(f"  Looking for sheet: {sheet_name}, column: {column_name}")

if __name__ == "__main__":
    main()
