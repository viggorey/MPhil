# Unified Workflow GUI

## Overview
The `master_workflow_gui.py` provides a unified interface for all steps in the ant tracking analysis workflow.

## Usage
Run the unified GUI:
```bash
python GUI/master_workflow_gui.py
```

## Workflow Steps
1. **Calibration** - Manage camera calibration sets
2. **Branch Sets** - Manage branch axis and radius calculations
3. **Datasets** - Link datasets to calibration and branch sets
4. **3D Reconstruction** - Reconstruct 3D coordinates from 2D tracking data
5. **Parameterization** - Calculate parameters from 3D data
6. **Analysis** - Statistical analysis of parameters
7. **Figures** - Generate publication-quality figures

## New Folder Structure
```
MPhil/
├── GUI/
│   └── master_workflow_gui.py
├── Processing/
│   ├── dlt_utils.py
│   ├── master_reconstruct.py
│   ├── master_parameterize.py
│   ├── parameterize_utils.py
│   ├── master_processing_gui.py
│   ├── master_analysis.py
│   ├── master_figures.py
│   └── master_analysis_figures_gui.py
├── Data/
│   ├── Calibration/
│   │   ├── calibration_sets.json
│   │   └── manage_calibration_sets_gui.py
│   ├── Branch_Sets/
│   │   └── branch_sets.json
│   ├── Datasets/
│   │   ├── dataset_links.json
│   │   ├── 2D_data/
│   │   ├── 3D_data/
│   │   └── 3D_data_params/
│   └── Videos/
│       └── manage_branch_sets_gui.py
└── Analysis/
    ├── results/
    └── figures/
```

## Migration Notes
- All 3D data files moved from `3D_reconstruction/` to `Data/Datasets/`
- Processing scripts moved to `Processing/` directory
- Analysis results moved to `Analysis/results/`
- All import paths have been updated
