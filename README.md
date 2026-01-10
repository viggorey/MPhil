# MPhil Project: Ant Locomotion Analysis Pipeline

Automated pipeline for 3D reconstruction, parameterization, statistical analysis, and figure generation for ant locomotion data.

## Overview

This project provides a comprehensive workflow for analyzing ant locomotion data, from 2D tracking data through 3D reconstruction, parameterization, statistical analysis, and publication-quality figure generation. The entire workflow is accessible through a unified graphical user interface.

## Project Structure

```
MPhil/
├── GUI/                          # Unified workflow GUI
│   ├── master_workflow_gui.py   # Main workflow interface
│   ├── export_import.py         # Data export/import functionality
│   └── README.md                # GUI-specific documentation
├── Processing/                   # Core processing modules
│   ├── master_reconstruct.py    # 3D reconstruction from 2D data
│   ├── master_parameterize.py   # Parameter calculation
│   ├── master_analysis.py       # Statistical analysis (ANOVA, post-hoc)
│   ├── master_figures.py        # Figure generation
│   ├── master_processing_gui.py  # Processing GUI (3D reconstruction & parameterization)
│   ├── master_analysis_figures_gui.py  # Analysis & figures GUI
│   ├── parameterize_utils.py    # Parameterization utilities
│   └── dlt_utils.py             # DLT (Direct Linear Transform) utilities
├── Data/                        # Data management
│   ├── Calibration/             # DLT calibration sets management
│   ├── Branch_Sets/            # Branch geometry data
│   ├── Datasets/               # Dataset storage
│   │   ├── 2D_data/           # 2D tracking data from Blender
│   │   ├── 3D_data/           # Reconstructed 3D coordinates
│   │   └── 3D_data_params/    # Parameterized datasets (Excel files)
│   ├── link_datasets_gui.py   # Dataset linking interface
│   └── dataset_links.json     # Dataset-to-calibration/branch links
├── Analysis/                    # Analysis results and figures
│   ├── results/                # Statistical analysis outputs
│   │   ├── [Metric]/          # Per-metric analysis results
│   │   └── master_analysis_summary.xlsx  # Summary spreadsheet
│   └── figures/                # Generated figures (PNG)
├── Config/                      # Configuration files
│   ├── species_data.json       # Species-specific data (CoM, leg joints)
│   └── processing_config.json  # Processing configuration
└── Blender scripts/            # Blender automation scripts
    ├── Video_Tracking/         # 2D tracking export/import scripts
    ├── CT_Analysis/          # CT scan analysis scripts
    └── README.md              # Blender scripts documentation
```

## Features

### Unified Workflow GUI

The main entry point is `GUI/master_workflow_gui.py`, which provides a single interface for all workflow steps:

1. **Calibration Management**: Create and manage DLT calibration sets
2. **Branch Set Management**: Calculate and store branch axis/radius data
3. **Dataset Linking**: Link ant datasets to calibration and branch sets
4. **3D Reconstruction**: Convert 2D tracking points to 3D coordinates
5. **Parameterization**: Calculate kinematic, biomechanical, and behavioral parameters
6. **Statistical Analysis**: Perform 2x2 factorial ANOVA with post-hoc analysis
7. **Figure Generation**: Generate publication-quality figures

### Data Export/Import

- Export all project data (2D, 3D, parameterized data, calibration sets, branch sets, dataset links) to a single JSON file
- Import data to restore project state
- Useful for backup and data transfer

### Statistical Analysis

- **2x2 Factorial ANOVA**: Analyzes effects of Species (Wax_Runner vs Non_Wax_Runner) and Substrate (Waxy vs Smooth)
- **Post-hoc Analysis**: Simple effects analysis with Holm-Bonferroni correction
- **Assumption Testing**: Normality and homogeneity of variance tests
- **Non-parametric Alternatives**: Aligned Rank Transform (ART) when assumptions are violated
- **Results Viewer**: Interactive viewing of statistical results with boxplots

### Figure Generation

- **Multi-Parameter Boxplots**: Generate boxplots for multiple parameters in a single figure
- **Single Parameter Boxplots**: Individual boxplots for each selected parameter
- **Gait Pattern Heatmap**: 2x2 heatmap showing leg attachment patterns across gait cycle
- **Feet-Attached Boxplots**: Boxplots filtered by number of attached feet
- **Consistent Styling**: Blue for Wax_Runner, green for Non_Wax_Runner, diagonal stripes for Waxy surfaces
- **Preview System**: View the 10 most recently generated figures

## Quick Start

### Running the Workflow GUI

```bash
python GUI/master_workflow_gui.py
```

This launches the unified workflow interface with all steps accessible from the sidebar.

### Workflow Steps

1. **Calibration**: Create DLT calibration sets from calibration videos
2. **Branch Sets**: Calculate branch axis and radius from branch videos
3. **Datasets**: Link ant tracking datasets to calibration and branch sets
4. **3D Reconstruction**: Reconstruct 3D coordinates from 2D tracking data
5. **Parameterization**: Calculate all parameters (kinematics, biomechanics, behavior)
6. **Analysis**: Run statistical analysis on selected metrics
7. **Figures**: Generate publication-quality figures

### Individual Scripts

If you prefer to run individual steps:

```bash
# 3D Reconstruction
python Processing/master_reconstruct.py

# Parameterization
python Processing/master_parameterize.py

# Statistical Analysis
python Processing/master_analysis.py

# Figure Generation
python Processing/master_figures.py
```

## Data Format

### Input Data

- **2D Tracking Data**: Excel files exported from Blender, containing tracked points for each frame
- **Calibration Sets**: JSON files containing DLT calibration parameters
- **Branch Sets**: JSON files containing branch axis, radius, and geometry data

### Output Data

- **3D Coordinates**: Excel files with reconstructed 3D coordinates for all tracked points
- **Parameterized Data**: Excel files with calculated parameters organized into sheets:
  - 3D_Coordinates
  - CoM (Center of Mass)
  - Speed
  - Duty_Factor
  - Kinematics
  - Biomechanics
  - Behavioral
  - Controls
  - Branch_Info
  - Size_Info
  - Trimming_Info

### Analysis Results

- **Per-Metric Results**: Text files with ANOVA results and boxplot images
- **Summary Spreadsheet**: Excel file with analysis summary for all metrics

## Parameters Calculated

### Kinematics
- Leg extension, orientation, and angles
- Footfall distances (longitudinal, lateral)
- CoM distances
- Speed and stride parameters

### Biomechanics
- Minimum Pull-Off Force
- Foot Plane Distance to CoM
- Cumulative Foot Spread
- L-distances (L_Distance_1 through L_Distance_5)

### Behavioral
- Duty factor
- Step frequency
- Gait patterns

### Body Measurements
- Body length
- Thorax length
- Leg lengths

## Statistical Analysis

The analysis performs:

1. **Normality Testing**: Shapiro-Wilk and Jarque-Bera tests
2. **Homogeneity Testing**: Levene's test
3. **Factorial ANOVA**: 2x2 design (Species × Substrate)
4. **Post-hoc Analysis**: Simple effects with Holm-Bonferroni correction
5. **Non-parametric Alternative**: ART when assumptions are violated

Results include:
- Main effects (Species, Substrate)
- Interaction effects
- Simple effects (substrate within species, species within substrate)
- Corrected p-values

## Figure Types

### Boxplots
- Consistent color scheme: Blue (#1f77b4) for Wax_Runner, Green (#2ca02c) for Non_Wax_Runner
- Diagonal stripes (///) for Waxy surfaces, solid fill for Smooth surfaces
- Individual data points overlaid
- Publication-ready formatting

### Gait Pattern Heatmap
- 2×2 grid showing leg attachment patterns
- Rows: Front/Middle/Hind legs
- Columns: Left/Right legs
- Color intensity represents attachment frequency

## Requirements

- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scipy
  - statsmodels
  - openpyxl
  - Pillow
  - tkinter (usually included with Python)

## Blender Integration

The project includes Blender scripts for:
- Exporting 2D tracking data to Excel
- Importing tracking data back to Blender
- CT scan analysis (CoM calculation, leg joint calculation)

See `Blender scripts/README.md` for detailed documentation.

## Notes

- All distance parameters are normalized by body length where appropriate
- Biomechanics parameters (Minimum Pull-Off Force) are NOT normalized
- The workflow supports both Wax_Runner (C. borneensis) and Non_Wax_Runner (C. captiosa) species
- Data is organized by condition codes: 11U (Wax_Runner_Waxy), 12U (Wax_Runner_Smooth), 21U (Non_Wax_Runner_Waxy), 22U (Non_Wax_Runner_Smooth)

## Troubleshooting

- **Permission Errors**: Ensure Excel files are closed before running parameterization
- **Missing Data**: Check that datasets are properly linked to calibration and branch sets
- **Empty Boxplots**: Verify that data exists for all 4 conditions (2 species × 2 substrates)
- **Analysis Skipped**: Check that sufficient data exists (minimum 2 observations per group)

## License

[Add your license information here]

## Contact

[Add your contact information here]
