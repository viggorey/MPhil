# Blender Scripts

This folder contains all Blender add-ons and scripts used for video tracking, data export, and anatomical measurements in the MPhil project.

## Folder Structure

```
Blender Scripts/
├── Video_Tracking/          # Export/import 2D tracking data from movie clips
│   ├── export_tracks_excel_combined.py
│   ├── export_tracks_clipboard.py
│   └── import_tracks_excel.py
│
└── CT_Analysis/             # Measurements from CT scan meshes
    ├── calculate_com.py
    ├── calculate_leg_joints.py
    └── copy_volume.py
```

---

## Video Tracking Scripts

Scripts for exporting and importing 2D tracking data from Blender movie clips.

### `export_tracks_excel_combined.py`

**Function**: Exports all tracks from a movie clip into a single Excel file with a structured format.

**Author**: Lorenz Mammen (GitHub @LorMam)

**Location**: Clip Editor > Side Panel: Track

**Usage**:
1. Open a movie clip in Blender's Clip Editor
2. Go to the "Track" panel in the sidebar
3. Select an output folder
4. Click "Export all Tracks"

**Output**:
- **Format**: Single Excel file containing all tracks
- **File naming**: `{clip_name}.xlsx` (e.g., `22U1L.xlsx`)
- **Excel structure**:
  - Sheet: "Track Data"
  - Columns: `Track Name`, `Coordinate Type`, `Frame {min_frame+1}`, `Frame {min_frame+2}`, ..., `Frame {max_frame-1}`
  - Rows: Two rows per track (one for X coordinates, one for Y coordinates)
  - Track naming: `Track 1`, `Track 2`, `Track 3`, etc.
  - **Note**: Tracks with fewer than 4 markers are exported as blank rows
  - **Note**: Frame 1 and the last frame are excluded from the output

**Example Output**:
```
Track Name | Coordinate Type | Frame 2 | Frame 3 | Frame 4 | ...
-----------|-----------------|----------|---------|---------|----
Track 1    | X               | 510.29   | 554.38  | 602.87  | ...
Track 1    | Y               | 758.28   | 418.82  | 39.68   | ...
Track 2    | X               | 727.42   | 779.22  | 843.14  | ...
Track 2    | Y               | 993.03   | 715.29  | 374.73  | ...
```

**Use case**: This format is compatible with the main processing pipeline (`Data/Datasets/2D_data/`).

---

### `export_tracks_clipboard.py`

**Function**: Copies track coordinates to the system clipboard instead of saving to file.

**Author**: Modified by AI Assistant

**Location**: Clip Editor > Side Panel: Track

**Usage**:
1. Open a movie clip in Blender's Clip Editor
2. Go to the "Track" panel in the sidebar
3. Click "Copy All Tracks" or "Copy Selected Track Only"

**Output**:
- **Format**: Plain text in clipboard
- **Content**: Comma-separated X,Y coordinates, one per line
- **Example**: 
  ```
  481.64, 993.03
  510.29, 758.28
  554.38, 418.82
  ```

**Use case**: Quick data transfer for testing or manual processing.

**Dependencies**: Requires `pyperclip` library (install via Blender's Python).

---

### `import_tracks_excel.py`

**Function**: Imports tracking data from an Excel file back into Blender.

**Author**: Lorenz Mammen (GitHub @LorMam)

**Location**: Clip Editor > Side Panel: Track

**Usage**:
1. Open a movie clip in Blender's Clip Editor
2. Go to the "Track" panel in the sidebar
3. Select an Excel file (must match the format from `export_tracks_excel_combined.py`)
4. Click "Import all Tracks"

**Input Format**:
- **Expected structure**: Same as `export_tracks_excel_combined.py` output
  - Columns: `Track Name`, `Coordinate Type`, `Frame {N}`, ...
  - Rows: Two rows per track (X and Y coordinates)
- **File format**: `.xlsx` or `.xls`

**Output**:
- Creates new tracks in the active movie clip
- Track names: Preserved from Excel file
- Markers: Created at the specified frame numbers with normalized coordinates (0-1 range)

**Use case**: Restoring tracking data, transferring tracks between projects, or correcting tracking data externally.

---

## CT Analysis Scripts

Scripts for extracting anatomical measurements from CT scan meshes.

### `calculate_com.py`

**Function**: Calculates the Center of Mass (CoM) of a mesh object and exports it along with two selected vertex coordinates to Excel.

**Author**: Viggo

**Location**: Mesh > Context Menu (right-click in Edit Mode)

**Usage**:
1. **Object Mode**: Select mesh object → Object → Set Origin → CoM (Volume)
   - This sets the object's origin to the volume-based center of mass
2. **Edit Mode**: Switch to Edit Mode
3. **Select vertices**: Shift + Left click to select exactly TWO vertices
4. **Export**: Right-click in 3D viewport → "Export Centroid and Vertex Coordinates"

**Output**:
- **Format**: Excel file
- **File location**: `/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/CoM/centroid_vertex_data.xlsx`
- **Excel structure**:
  - Sheet: "Centroid and Vertex Data"
  - Columns: `Object Name`, `Centroid X`, `Centroid Y`, `Centroid Z`, `Vertex 1 X`, `Vertex 1 Y`, `Vertex 1 Z`, `Vertex 2 X`, `Vertex 2 Y`, `Vertex 2 Z`
  - Rows: One row per export operation (data is appended)
- **Coordinate system**: 
  - **Centroid (CoM)**: Always `(0, 0, 0)` because the object origin is set to CoM
  - **Vertices**: World coordinates of the two selected vertices

**Example Output**:
```
Object Name | Centroid X | Centroid Y | Centroid Z | Vertex 1 X | Vertex 1 Y | Vertex 1 Z | Vertex 2 X | Vertex 2 Y | Vertex 2 Z
------------|------------|------------|------------|------------|------------|------------|------------|------------|------------
Ant_Head    | 0.0        | 0.0        | 0.0        | 1.234      | 0.567      | -0.123     | 1.456      | 0.789      | -0.234
```

**Use case**: Extracting CoM data and reference points for use in `Config/species_data.json` for the parameterization pipeline.

**Note**: The script preserves vertex selection order when possible, but may fall back to index order if selection history is incomplete.

---

### `calculate_leg_joints.py`

**Function**: Exports selected vertices in their selection order to Excel for leg joint position extraction.

**Author**: Viggo

**Location**: Mesh > Context Menu (right-click in Edit Mode)

**Usage**:
1. **Edit Mode**: Switch to Edit Mode
2. **Select vertices**: Shift + Left click to select vertices in the desired order
   - **Important**: Selection order matters! Vertices are exported in the order they were selected.
   - Maximum: 100 vertices
3. **Export**: Right-click in 3D viewport → "Export Vertex Positions"

**Output**:
- **Format**: Excel file
- **File location**: `/Users/viggorey/Desktop/PhD/Cambridge/Macaranga/CoM/Leg_joints.xlsx`
- **Excel structure**:
  - Sheet: "Vertex Positions"
  - Columns: `Vertex Order`, `X`, `Y`, `Z`
  - Rows: One row per selected vertex (in selection order)
- **Coordinate system**: World coordinates of selected vertices

**Example Output**:
```
Vertex Order | X      | Y      | Z
-------------|--------|--------|--------
1            | 1.234  | 0.567  | -0.123
2            | 1.456  | 0.789  | -0.234
3            | 1.678  | 0.901  | -0.345
```

**Use case**: Extracting leg joint positions from CT scan data for use in leg joint interpolation calculations during 3D reconstruction.

**Important Notes**:
- **Selection order is critical**: The script requires complete selection history to determine order. If selection history is incomplete, the export will fail with a warning.
- **Point mapping**: In the trim code, point 1 corresponds to point 2, and point 5 corresponds to point 3 (as noted in the script comments).

---

### `copy_volume.py`

**Function**: Calculates the volume of a selected mesh object and copies the result to the system clipboard.

**Author**: Your Name

**Location**: 3D Viewport > Sidebar (N) > Tool Tab

**Usage**:
1. Select a mesh object in the 3D viewport
2. Press `N` to open the sidebar (if not already visible)
3. Go to the "Tool" tab
4. Find the "Volume Calculator" panel
5. Click "Calculate Volume"
6. Volume is automatically copied to clipboard
7. Check the panel for the result

**Output**:
- **Format**: Plain text in clipboard
- **Content**: Volume value formatted to 6 decimal places (e.g., `1.234567`)
- **Units**: Blender units³ (based on scene unit settings)
- **Display**: Also shown in the panel's "Last Result" section

**Requirements**:
- Object must be a **manifold (watertight) mesh** for accurate volume calculation
- Works in Object mode (automatically switches if needed)
- Accounts for object transforms (scale, rotation, position)

**Dependencies**:
- **Linux users**: May need to install `xclip` or `xsel`:
  - Ubuntu/Debian: `sudo apt install xclip`
  - Fedora: `sudo dnf install xclip`

**Use case**: Quick volume measurements for CT scan analysis or mesh validation.

---

## Installation

### Installing Add-ons in Blender

1. Open Blender
2. Go to **Edit → Preferences → Add-ons**
3. Click **"Install..."** button
4. Navigate to the appropriate folder (`Video_Tracking/` or `CT_Analysis/`)
5. Select the `.py` file you want to install
6. Enable the add-on in the list by checking the checkbox
7. The add-on will now be available in the appropriate Blender menu

### Installing Python Dependencies

Most scripts require `openpyxl` for Excel file handling.

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

**For clipboard functionality** (`export_tracks_clipboard.py`):
```bash
# Install pyperclip (use Blender's Python as shown above)
python -m pip install pyperclip
```

---

## Workflow Integration

These scripts are part of the MPhil project pipeline:

1. **Video Tracking** (`Video_Tracking/export_tracks_excel_combined.py`):
   - Export 2D tracking data from Blender → `Data/Datasets/2D_data/`
   - Format: Single Excel file per camera view (Left, Right, Top, Front)

2. **CoM Calculation** (`CT_Analysis/calculate_com.py`):
   - Extract CoM data from CT scans → Used in `Config/species_data.json`
   - Provides reference points for body segment CoM calculations

3. **Leg Joint Measurement** (`CT_Analysis/calculate_leg_joints.py`):
   - Extract leg joint positions → Used in leg joint interpolation during 3D reconstruction
   - Provides anatomical reference points for scaling and positioning

---

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

### Clipboard not working:
- **Linux**: Install `xclip` or `xsel` (see Dependencies section)
- **Windows/macOS**: Should work out of the box
- For `pyperclip`: Ensure it's installed in Blender's Python environment

---

## Version Compatibility

- **Blender**: 2.8+ (most scripts), 3.0+ (CT Analysis scripts)
- **Python**: 3.7+ (Blender's bundled Python)
- **openpyxl**: Latest version recommended
- **pyperclip**: Latest version (for clipboard export)

---

## Support

For issues or questions about these scripts, refer to:
- Blender API documentation: https://docs.blender.org/api/current/
- Original author credits in script headers
- Project documentation in main `README.md`
