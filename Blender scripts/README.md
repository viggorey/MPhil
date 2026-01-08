# Blender Scripts

This folder contains all Blender add-ons and scripts used for video tracking, data export, and anatomical measurements in the MPhil project.

## Overview

These scripts are used throughout the data processing pipeline:
1. **Video Tracking**: Export 2D tracking data from Blender movie clips
2. **CoM Calculation**: Calculate Center of Mass from CT scan meshes
3. **Leg Joint Measurement**: Extract leg joint positions from CT scan data

## Export Scripts

### `export_add_on.py`
Main Blender add-on for exporting video tracking data. Exports all tracks from all movie clips to CSV format.

**Author**: Lorenz Mammen (GitHub @LorMam)

**Usage**: 
- Install as Blender add-on
- Use from Clip Editor > Side Panel: Track
- Exports tracked xy coordinates to `.csv` files

**API Reference**: `bpy.data.movieclips[0].tracking.tracks[0].markers[0].co`

### `export_add_on names.py`
Variant of export add-on that includes track names in the export. Useful when you need to identify which track corresponds to which body part.

### `export_add_1excel.py`
Exports tracking data to Excel format (single file). All tracks are combined into one Excel workbook with multiple sheets.

### `export_add_1excel_picture.py`
Exports tracking data to Excel format with picture references. Includes thumbnail images of the tracked points.

### `export_copy.py`
Copies tracking data to clipboard instead of saving to file. Useful for quick data transfer or testing.

### `import_add_1excel.py`
Imports tracking data from Excel file back into Blender. Allows you to restore or transfer tracking data between projects.

## CoM and Measurement Scripts

### `CoM_Calc.py`
Blender add-on for calculating and exporting Center of Mass (CoM) coordinates from CT scan meshes.

**Author**: Viggo

**Usage**:
1. In Object mode: Object → Set Origin → CoM (Volume)
2. Go to Edit mode
3. Shift + Left click to select TWO vertices
4. Right click in 3D viewport → "Export Centroid and Vertex Coordinates"
5. CoM = 0, 0, 0; rest are valid coordinates

**Output**: Excel file with CoM coordinates and selected vertex coordinates

### `Leg_Joint_Calc.py`
Blender add-on for calculating and exporting leg joint positions from CT scan meshes. Used to extract anatomical reference points for leg joint interpolation.

### `Copy_volume.py`
Utility script for copying volume data to clipboard. Used for quick data transfer during CT scan analysis.

## Installation

To install these as Blender add-ons:

1. Open Blender
2. Go to **Edit → Preferences → Add-ons**
3. Click **"Install..."** button
4. Navigate to this folder and select the `.py` file you want to install
5. Enable the add-on in the list by checking the checkbox
6. The add-on will now be available in the appropriate Blender menu

## Dependencies

### Required Python Package:
- **`openpyxl`** - For Excel file handling

### Installation in Blender's Python:

**macOS:**
```bash
/Applications/Blender.app/Contents/Resources/4.2/python/bin/python3.11 -m pip install openpyxl
```

**Windows:**
```bash
C:\Program Files\Blender Foundation\Blender 4.2\4.2\python\bin\python.exe -m pip install openpyxl
```

**Linux:**
```bash
/usr/bin/blender --python-expr "import subprocess; subprocess.call(['pip', 'install', 'openpyxl'])"
```

Or use Blender's built-in Python console:
```python
import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'install', 'openpyxl'])
```

## Workflow Integration

These scripts are part of the MPhil project pipeline:

1. **Video Tracking** (`export_add_1excel.py`):
   - Export 2D tracking data from Blender → `Data/Videos/2D_data/`

2. **CoM Calculation** (`CoM_Calc.py`):
   - Extract CoM data from CT scans → Used in `Config/species_data.json`

3. **Leg Joint Measurement** (`Leg_Joint_Calc.py`):
   - Extract leg joint positions → Used in parameterization calculations

## File Organization

All Blender scripts have been consolidated in this folder for easier management. The original files from:
- `Blender export/` folder (deleted)
- `CoM/Code/` folder (deleted)
- `3D transformation/4. Code/Blender export/` folder (duplicates, deleted)

are now located here.

## Troubleshooting

### Add-on not appearing in Blender:
- Check that the file has `.py` extension
- Verify the `bl_info` dictionary is properly formatted
- Check Blender's console for error messages (Window → Toggle System Console)

### Excel export fails:
- Ensure `openpyxl` is installed in Blender's Python environment
- Check file permissions for the output directory
- Verify Blender version compatibility (most scripts support Blender 2.8+)

### Import errors:
- Make sure all dependencies are installed
- Check Python version compatibility
- Review Blender's console for specific error messages

## Version Compatibility

- **Blender**: 2.8+ (most scripts), 3.0+ (CoM_Calc.py)
- **Python**: 3.7+ (Blender's bundled Python)
- **openpyxl**: Latest version recommended

## Support

For issues or questions about these scripts, refer to:
- Blender API documentation: https://docs.blender.org/api/current/
- Original author credits in script headers
- Project documentation in main `README.md`
