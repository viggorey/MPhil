# MPhil Project: Ant Locomotion Analysis Pipeline

Automated pipeline for 3D reconstruction, parameterization, and analysis of ant locomotion data.

## Project Structure

```
MPhil/
├── Data/                    # Input data and configuration
│   ├── Calibration/         # DLT calibration sets
│   └── Videos/              # Video data and branch sets
├── 3D_reconstruction/       # 3D reconstruction and parameterization
│   ├── 3D_data/            # Reconstructed 3D datasets
│   └── 3D_data_params/     # Parameterized datasets
├── Analysis/               # Statistical analysis and figures
│   ├── analysisresults/    # Analysis outputs
│   └── Figures/            # Generated figures
├── Config/                 # Configuration files
│   ├── species_data.json   # Species-specific data (CoM, leg joints)
│   └── processing_config.json
└── Blender scripts/        # Blender automation scripts
```

## Workflow

1. **Calibration Management**: Use `Data/Calibration/manage_calibration_sets.py` to create and manage DLT calibration sets
2. **Branch Data Management**: Use `Data/Videos/manage_branch_sets.py` to calculate and store branch axis/radius
3. **Dataset Linking**: Use `Data/link_datasets.py` to link ant datasets to calibration and branch sets
4. **3D Reconstruction**: Run `3D_reconstruction/master_reconstruct.py` to reconstruct 2D data to 3D
5. **Parameterization**: Run `3D_reconstruction/master_parameterize.py` to calculate all parameters
6. **Analysis**: Run `Analysis/master_analysis.py` for statistical analysis
7. **Figures**: Run `Analysis/master_figures.py` to generate thesis figures

## Quick Start

See individual script documentation for usage instructions.
