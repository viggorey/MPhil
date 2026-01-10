"""
GUI tool to manage branch sets.
Create, update, delete, and view branch sets with branch axis/radius calculation.
"""

import json
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path to import dlt_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Processing"))
from dlt_utils import reconstruct_3d_point

BRANCH_SETS_FILE = Path(__file__).parent.parent / "Branch_Sets" / "branch_sets.json"
CALIBRATION_SETS_FILE = Path(__file__).parent.parent / "Calibration" / "calibration_sets.json"
DEFAULT_CAMERAS = ["Left", "Top", "Right", "Front"]


def load_branch_sets():
    """Load branch sets from JSON file."""
    if BRANCH_SETS_FILE.exists():
        try:
            with open(BRANCH_SETS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading branch_sets.json: {e}")
            print(f"File may be corrupted. Creating backup and starting fresh.")
            # Create backup
            backup_file = BRANCH_SETS_FILE.with_suffix('.json.bak')
            if BRANCH_SETS_FILE.exists():
                import shutil
                shutil.copy2(BRANCH_SETS_FILE, backup_file)
                print(f"Backup saved to: {backup_file}")
            return {}
    return {}


def save_branch_sets(branch_sets):
    """Save branch sets to JSON file."""
    BRANCH_SETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Write to a temporary file first, then rename (atomic write)
        temp_file = BRANCH_SETS_FILE.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(branch_sets, f, indent=2, ensure_ascii=False)
        # Replace the original file
        if temp_file.exists():
            if BRANCH_SETS_FILE.exists():
                BRANCH_SETS_FILE.unlink()
            temp_file.replace(BRANCH_SETS_FILE)
    except Exception as e:
        # If something goes wrong, try direct write as fallback
        print(f"Warning: Could not use atomic write, using direct write: {e}")
        with open(BRANCH_SETS_FILE, 'w') as f:
            json.dump(branch_sets, f, indent=2, ensure_ascii=False)


def load_calibration_sets():
    """Load calibration sets from JSON file."""
    if CALIBRATION_SETS_FILE.exists():
        with open(CALIBRATION_SETS_FILE, 'r') as f:
            return json.load(f)
    return {}


def reconstruct_branch_axis(A, points_2d):
    """
    Reconstruct 3D branch axis from 2D points in multiple camera views.
    EXACTLY matches D1_Metadata.py implementation.
    
    Parameters:
    -----------
    A : np.ndarray
        Camera coefficients matrix, shape (11, n_cameras)
    points_2d : list of np.ndarray
        List of 2D points arrays, one per camera (in same order as A columns).
        Empty arrays for cameras without points.
    """
    from scipy import stats, linalg
    
    def get_direction_from_points(points):
        """Calculate the primary direction of a set of 2D points."""
        x = points[:, 0]
        y = points[:, 1]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        direction = np.array([1.0, slope])
        return direction / np.linalg.norm(direction)
    
    # Filter out empty arrays and get valid camera indices
    # This matches D1_Metadata exactly: valid_cameras = [i for i, points in enumerate(points_2d) if points.size > 0]
    valid_cameras = [i for i, points in enumerate(points_2d) if points.size > 0]
    if len(valid_cameras) < 2:
        raise ValueError("Need at least 2 cameras with points for reconstruction")
    
    # Use only valid cameras and their corresponding coefficients
    A_subset = A[:, valid_cameras]
    n_cameras = len(valid_cameras)
    midline_vectors_2d = np.zeros((n_cameras, 2))
    
    for i, cam_idx in enumerate(valid_cameras):
        direction = get_direction_from_points(points_2d[cam_idx])
        midline_vectors_2d[i] = direction
    
    M = np.zeros((2 * n_cameras, 3 + n_cameras))
    
    for i in range(n_cameras):
        L = A_subset[:, i]
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
    
    # Check if we need to flip the direction to align with y-axis
    if abs(direction[1]) > abs(direction[0]) and abs(direction[1]) > abs(direction[2]):
        if direction[1] < 0:  # If y component is negative, flip the vector
            direction = -direction
    
    # Use first valid camera's points for centroid calculation
    centroid_2d = np.mean(points_2d[valid_cameras[0]], axis=0)
    
    P = np.zeros((2 * n_cameras, 3))
    b = np.zeros(2 * n_cameras)
    
    for i, cam_idx in enumerate(valid_cameras):
        L = A[:, cam_idx]
        u, v = np.mean(points_2d[cam_idx], axis=0)
        
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


def calculate_branch_radius(A, axis_direction, axis_point, surface_points_2d, camera_list):
    """
    Calculate branch radius from surface points.
    Based on D1_Metadata.py create_branch_cylinder function.
    """
    # Reconstruct surface points to 3D
    surface_points_3d = []
    
    # Get valid cameras
    valid_cameras = [i for i, cam_name in enumerate(camera_list) 
                     if cam_name in surface_points_2d and len(surface_points_2d[cam_name]) > 0]
    
    if len(valid_cameras) < 2:
        raise ValueError("Need at least 2 cameras for surface point reconstruction")
    
    A_subset = A[:, valid_cameras]
    
    # Find the minimum number of points across all cameras (to handle mismatched counts)
    n_points_per_camera = [len(surface_points_2d[camera_list[cam_idx]]) for cam_idx in valid_cameras]
    n_points = min(n_points_per_camera) if n_points_per_camera else 0
    
    if n_points < 2:
        raise ValueError("Need at least 2 surface points per camera to calculate radius")
    
    # Reconstruct each surface point (each point is seen from multiple cameras)
    for point_idx in range(n_points):
        point_2d_list = []
        for cam_idx in valid_cameras:
            cam_name = camera_list[cam_idx]
            if point_idx < len(surface_points_2d[cam_name]):
                point_2d_list.append(np.array(surface_points_2d[cam_name][point_idx]))
            else:
                # Skip this point if not available in this camera
                continue
        
        # Need at least 2 cameras to reconstruct
        if len(point_2d_list) >= 2:
            try:
                # point_2d_list is already in the same order as valid_cameras
                # reconstruct_3d_point expects a list of 2D points, one per camera in A_subset
                point_3d = reconstruct_3d_point(A_subset, point_2d_list)
                surface_points_3d.append(point_3d)
            except Exception as e:
                print(f"    Warning: Could not reconstruct surface point {point_idx}: {e}")
                continue
    
    if len(surface_points_3d) < 2:
        raise ValueError("Need at least 2 successfully reconstructed surface points to calculate radius")
    
    # Calculate radius using the same method as D1_Metadata create_branch_cylinder
    # This matches the original implementation exactly
    axis_direction = np.array(axis_direction)
    axis_point = np.array(axis_point)
    surface_points_3d = np.array(surface_points_3d)
    
    # Normalize axis direction
    z_axis = axis_direction / np.linalg.norm(axis_direction)
    
    # Get relative points (from axis point)
    relative_points = surface_points_3d - axis_point
    
    # Project points to get radii at different positions
    # Remove the component along the axis, leaving only perpendicular component
    projected_points = relative_points - np.outer(np.dot(relative_points, z_axis), z_axis)
    
    # Calculate radii (distance from axis)
    radii = np.linalg.norm(projected_points, axis=1)
    
    # Mean radius (matches D1_Metadata exactly)
    mean_radius = np.mean(radii)
    
    return float(mean_radius)


class BranchSetManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Branch Set Manager")
        self.root.geometry("1200x800")
        
        self.branch_sets = load_branch_sets()
        self.calibration_sets = load_calibration_sets()
        self.camera_list = DEFAULT_CAMERAS.copy()
        self.current_calibration_set = None
        self.current_calibration_coefficients = None
        self.current_results = {}  # Store calculated results
        self.axis_camera_enabled = {cam: True for cam in DEFAULT_CAMERAS}  # Track which cameras are enabled for axis
        self.surface_camera_enabled = {cam: True for cam in DEFAULT_CAMERAS}  # Track which cameras are enabled for surface
        
        # Main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        # Left panel - List of branch sets
        left_frame = ttk.LabelFrame(main_frame, text="Branch Sets", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.set_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=15)
        self.set_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.set_listbox.yview)
        
        self.set_listbox.bind('<<ListboxSelect>>', self.on_set_select)
        
        # Buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="New", command=self.create_new).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Delete", command=self.delete_set).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Refresh", command=self.refresh_list).pack(side=tk.LEFT)
        
        # Right panel - Details/Editor
        right_frame = ttk.LabelFrame(main_frame, text="Branch Set Details", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Name field
        name_frame = ttk.Frame(right_frame)
        name_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(name_frame, text="Name:").pack(side=tk.LEFT)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(name_frame, textvariable=self.name_var, width=30)
        self.name_entry.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        
        # Calibration set selector
        cal_frame = ttk.Frame(right_frame)
        cal_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(cal_frame, text="Calibration Set:").pack(side=tk.LEFT)
        self.cal_var = tk.StringVar()
        self.cal_combo = ttk.Combobox(cal_frame, textvariable=self.cal_var, width=27, state="readonly")
        self.cal_combo.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        self.cal_combo.bind('<<ComboboxSelected>>', self.on_calibration_selected)
        self.refresh_calibration_list()
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Tab: Results (create before camera tabs)
        self.tab_results = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_results, text="Results")
        self.setup_results_tab()
        
        # Camera tabs (created dynamically)
        self.camera_tabs = {}
        self.camera_tables = {}
        self.surface_tabs = {}
        self.surface_tables = {}
        self.refresh_camera_tabs()
        
        # Save and Calculate buttons
        btn_save_frame = ttk.Frame(right_frame)
        btn_save_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btn_save_frame, text="Save Branch Set", command=self.save_set).pack(side=tk.LEFT)
        ttk.Button(btn_save_frame, text="Calculate Axis & Radius", command=self.calculate_branch).pack(side=tk.LEFT, padx=(10, 0))
        
        self.refresh_list()
        self.current_set_name = None
    
    def refresh_camera_tabs(self):
        """Refresh camera tabs based on current camera list."""
        # Remove existing tabs
        for camera in list(self.camera_tabs.keys()):
            if camera not in self.camera_list:
                # Remove branch axis tab
                tab = self.camera_tabs[camera]
                self.notebook.forget(tab)
                del self.camera_tabs[camera]
                if camera in self.camera_tables:
                    del self.camera_tables[camera]
                # Remove surface tab
                if camera in self.surface_tabs:
                    tab = self.surface_tabs[camera]
                    self.notebook.forget(tab)
                    del self.surface_tabs[camera]
                if camera in self.surface_tables:
                    del self.surface_tables[camera]
                # Remove from enabled dicts
                if camera in self.axis_camera_enabled:
                    del self.axis_camera_enabled[camera]
                if camera in self.surface_camera_enabled:
                    del self.surface_camera_enabled[camera]
        
        # Add/update camera tabs
        for camera in self.camera_list:
            # Initialize enabled state if not present
            if camera not in self.axis_camera_enabled:
                self.axis_camera_enabled[camera] = True
            if camera not in self.surface_camera_enabled:
                self.surface_camera_enabled[camera] = True
            
            # Branch axis points tab
            if camera not in self.camera_tabs:
                tab = ttk.Frame(self.notebook, padding="10")
                self.notebook.insert(self.notebook.index(self.tab_results), tab, text=f"{camera} Axis")
                self.camera_tabs[camera] = tab
                self.setup_camera_tab(camera, tab, is_axis=True)
            else:
                tab = self.camera_tabs[camera]
                idx = self.notebook.index(tab)
                self.notebook.tab(idx, text=f"{camera} Axis")
            
            # Surface points tab
            if camera not in self.surface_tabs:
                tab = ttk.Frame(self.notebook, padding="10")
                self.notebook.insert(self.notebook.index(self.tab_results), tab, text=f"{camera} Surface")
                self.surface_tabs[camera] = tab
                self.setup_camera_tab(camera, tab, is_axis=False)
            else:
                tab = self.surface_tabs[camera]
                idx = self.notebook.index(tab)
                self.notebook.tab(idx, text=f"{camera} Surface")
    
    def on_axis_camera_toggle(self, camera, enabled):
        """Handle axis camera enable/disable toggle."""
        self.axis_camera_enabled[camera] = enabled
    
    def on_surface_camera_toggle(self, camera, enabled):
        """Handle surface camera enable/disable toggle."""
        self.surface_camera_enabled[camera] = enabled
    
    def setup_camera_tab(self, camera, tab, is_axis=True):
        """Setup a camera's points input tab (axis or surface)."""
        point_type = "axis" if is_axis else "surface"
        instructions_text = f"Enter 2D {point_type} points in format: [u, v] (one per line) for {camera} camera."
        if is_axis:
            instructions_text += " These are points along the branch centerline."
        else:
            instructions_text += " These are points on the branch surface edges."
        
        # Top frame with instructions and enable checkbox
        top_frame = ttk.Frame(tab)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        instructions = ttk.Label(top_frame, text=instructions_text, font=('TkDefaultFont', 9))
        instructions.pack(side=tk.LEFT, anchor=tk.W)
        
        # Enable/disable checkbox
        enable_var = tk.BooleanVar(value=True)
        if is_axis:
            if camera not in self.axis_camera_enabled:
                self.axis_camera_enabled[camera] = True
            enable_var.set(self.axis_camera_enabled[camera])
            enable_var.trace('w', lambda *args, cam=camera: self.on_axis_camera_toggle(cam, enable_var.get()))
        else:
            if camera not in self.surface_camera_enabled:
                self.surface_camera_enabled[camera] = True
            enable_var.set(self.surface_camera_enabled[camera])
            enable_var.trace('w', lambda *args, cam=camera: self.on_surface_camera_toggle(cam, enable_var.get()))
        
        checkbox = ttk.Checkbutton(top_frame, text="Use in calculation", variable=enable_var)
        checkbox.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Store the variable for later access
        if is_axis:
            setattr(self, f'axis_enable_{camera}', enable_var)
        else:
            setattr(self, f'surface_enable_{camera}', enable_var)
        
        # Text input area
        text_frame = ttk.Frame(tab)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        text_widget = scrolledtext.ScrolledText(text_frame, height=12, width=40, font=('Courier', 10))
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder
        placeholder = "[0.0, 0.0]\n[0.0, 0.0]\n[0.0, 0.0]"
        text_widget.insert('1.0', placeholder)
        text_widget.placeholder_set = True
        
        def on_focus_in(event):
            if text_widget.placeholder_set:
                text_widget.delete('1.0', tk.END)
                text_widget.placeholder_set = False
        
        text_widget.bind('<FocusIn>', on_focus_in)
        
        if is_axis:
            self.camera_tables[camera] = text_widget
        else:
            self.surface_tables[camera] = text_widget
        
        # Button frame
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X)
        
        def load_from_text():
            self.load_2d_from_text(camera, is_axis)
        
        def clear_text():
            text_widget.delete('1.0', tk.END)
            text_widget.insert('1.0', placeholder)
            text_widget.placeholder_set = True
            tree = getattr(self, f'tree_2d_{camera}_{point_type}', None)
            if tree:
                for item in tree.get_children():
                    tree.delete(item)
        
        def save_points():
            self.save_camera_points(camera, is_axis)
        
        def load_points():
            self.load_camera_points(camera, is_axis)
        
        ttk.Button(btn_frame, text="Load from Text", command=load_from_text).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text=f"Save {camera} {point_type.title()} Points", command=save_points).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text=f"Load {camera} {point_type.title()} Points", command=load_points).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Clear", command=clear_text).pack(side=tk.LEFT, padx=(0, 5))
        
        # Table frame
        table_label = ttk.Label(tab, text="Parsed Points:", font=('TkDefaultFont', 9, 'bold'))
        table_label.pack(anchor=tk.W, pady=(10, 5))
        
        table_frame = ttk.Frame(tab)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Point', 'U', 'V')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        setattr(self, f'tree_2d_{camera}_{point_type}', tree)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_results_tab(self):
        """Setup the results display tab."""
        self.results_text = scrolledtext.ScrolledText(self.tab_results, height=20, width=60)
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def refresh_list(self):
        """Refresh the list of branch sets."""
        self.set_listbox.delete(0, tk.END)
        for name in sorted(self.branch_sets.keys()):
            self.set_listbox.insert(tk.END, name)
    
    def refresh_calibration_list(self):
        """Refresh the calibration set dropdown."""
        cal_names = sorted(self.calibration_sets.keys())
        self.cal_combo['values'] = cal_names
        if cal_names and not self.cal_var.get():
            self.cal_var.set(cal_names[0])
            self.on_calibration_selected()
    
    def on_calibration_selected(self, event=None):
        """Handle calibration set selection."""
        cal_name = self.cal_var.get()
        if cal_name and cal_name in self.calibration_sets:
            self.current_calibration_set = cal_name
            cal_set = self.calibration_sets[cal_name]
            
            # Extract camera coefficients
            # Check if cameras_list exists, otherwise use default
            camera_list = cal_set.get('cameras_list', DEFAULT_CAMERAS)
            if camera_list != self.camera_list:
                self.camera_list = camera_list.copy()
                self.refresh_camera_tabs()
            
            # Build A matrix (11 x n_cameras)
            cameras_data = cal_set.get('cameras', {})
            coefficients_list = []
            for camera in self.camera_list:
                if camera in cameras_data and 'coefficients' in cameras_data[camera]:
                    coefficients_list.append(cameras_data[camera]['coefficients'])
                else:
                    messagebox.showwarning("Warning", f"No coefficients found for {camera} camera in {cal_name}")
                    return
            
            if coefficients_list:
                self.current_calibration_coefficients = np.array(coefficients_list).T  # Shape: (11, n_cameras)
    
    def on_set_select(self, event):
        """Handle selection of a branch set."""
        selection = self.set_listbox.curselection()
        if not selection:
            return
        
        set_name = self.set_listbox.get(selection[0])
        self.load_set(set_name)
    
    def parse_2d_text(self, text):
        """Parse 2D points from text format [u, v]."""
        points = []
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                line = line.strip('[]')
                coords = [float(x.strip()) for x in line.split(',')]
                if len(coords) == 2:
                    points.append(coords)
            except (ValueError, AttributeError):
                continue
        return np.array(points) if points else None
    
    def load_2d_from_text(self, camera, is_axis=True, show_message=True):
        """Load 2D points from text area into table."""
        point_type = "axis" if is_axis else "surface"
        text_widget = self.camera_tables[camera] if is_axis else self.surface_tables[camera]
        text = text_widget.get('1.0', tk.END)
        points = self.parse_2d_text(text)
        
        if points is None or len(points) == 0:
            if show_message:
                messagebox.showerror("Error", f"Could not parse 2D {point_type} points. Use format: [u, v] (one per line)")
            return
        
        tree = getattr(self, f'tree_2d_{camera}_{point_type}', None)
        if tree is None:
            return
        
        # Clear table
        for item in tree.get_children():
            tree.delete(item)
        
        # Add points to table
        for i, point in enumerate(points, 1):
            tree.insert('', tk.END, values=(i, point[0], point[1]))
        
        if show_message:
            messagebox.showinfo("Success", f"Loaded {len(points)} {point_type} points for {camera} camera")
    
    def save_camera_points(self, camera, is_axis=True):
        """Save 2D points for a camera to a file."""
        from tkinter import filedialog
        point_type = "axis" if is_axis else "surface"
        text_widget = self.camera_tables[camera] if is_axis else self.surface_tables[camera]
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title=f"Save {camera} {point_type.title()} Points"
        )
        if filename:
            text = text_widget.get('1.0', tk.END)
            with open(filename, 'w') as f:
                f.write(text)
            messagebox.showinfo("Success", f"{camera} {point_type} points saved to {filename}")
    
    def load_camera_points(self, camera, is_axis=True):
        """Load 2D points for a camera from a file."""
        from tkinter import filedialog
        point_type = "axis" if is_axis else "surface"
        text_widget = self.camera_tables[camera] if is_axis else self.surface_tables[camera]
        filename = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title=f"Load {camera} {point_type.title()} Points"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    text = f.read()
                text_widget.delete('1.0', tk.END)
                text_widget.insert('1.0', text)
                text_widget.placeholder_set = False
                self.load_2d_from_text(camera, is_axis)
                messagebox.showinfo("Success", f"{camera} {point_type} points loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")
    
    def get_2d_points(self, camera, is_axis=True):
        """Get 2D points for a camera from the table."""
        point_type = "axis" if is_axis else "surface"
        tree = getattr(self, f'tree_2d_{camera}_{point_type}', None)
        if tree is None:
            return None
        points = []
        for item in tree.get_children():
            values = tree.item(item, 'values')
            points.append([float(values[1]), float(values[2])])
        return np.array(points) if points else None
    
    def calculate_branch(self):
        """Calculate branch axis and radius."""
        if self.current_calibration_coefficients is None:
            messagebox.showerror("Error", "Please select a calibration set first.")
            return
        
        # Collect branch axis points (only from enabled cameras)
        # Build as a LIST in camera_list order (matching A matrix columns), with empty arrays for disabled/missing cameras
        branch_points_2d_list = []
        enabled_axis_indices = []
        for i, camera in enumerate(self.camera_list):
            if self.axis_camera_enabled.get(camera, True):  # Default to True if not set
                points = self.get_2d_points(camera, is_axis=True)
                if points is not None and len(points) >= 2:
                    branch_points_2d_list.append(points)
                    enabled_axis_indices.append(i)
                else:
                    branch_points_2d_list.append(np.array([]))  # Empty array for missing points
            else:
                branch_points_2d_list.append(np.array([]))  # Empty array for disabled cameras
        
        if len(enabled_axis_indices) < 2:
            messagebox.showerror("Error", f"Need branch axis points from at least 2 enabled cameras. Currently have {len(enabled_axis_indices)} enabled camera(s) with valid points.")
            return
        
        # Collect surface points (only from enabled cameras) - keep as dict for radius calculation
        surface_points_2d = {}
        enabled_surface_cameras = []
        for camera in self.camera_list:
            if self.surface_camera_enabled.get(camera, True):  # Default to True if not set
                points = self.get_2d_points(camera, is_axis=False)
                if points is not None and len(points) >= 2:
                    surface_points_2d[camera] = points
                    enabled_surface_cameras.append(camera)
        
        if len(surface_points_2d) < 2:
            messagebox.showwarning("Warning", f"Surface points from fewer than 2 enabled cameras ({len(enabled_surface_cameras)}). Radius calculation may fail.")
        
        # Reconstruct branch axis - pass as LIST to match D1_Metadata signature
        try:
            axis_direction, axis_point = reconstruct_branch_axis(
                self.current_calibration_coefficients, 
                branch_points_2d_list  # Pass as list, not dict
            )
            # Convert numpy arrays to lists for storage
            if isinstance(axis_direction, np.ndarray):
                self.current_results['axis_direction'] = axis_direction.tolist()
            else:
                self.current_results['axis_direction'] = axis_direction
            if isinstance(axis_point, np.ndarray):
                self.current_results['axis_point'] = axis_point.tolist()
            else:
                self.current_results['axis_point'] = axis_point
        except Exception as e:
            messagebox.showerror("Error", f"Could not reconstruct branch axis: {e}")
            return
        
        # Calculate radius
        radius = None
        if len(surface_points_2d) >= 2:
            try:
                radius = calculate_branch_radius(
                    self.current_calibration_coefficients,
                    axis_direction,
                    axis_point,
                    surface_points_2d,
                    self.camera_list
                )
                self.current_results['radius'] = radius
            except Exception as e:
                messagebox.showwarning("Warning", f"Could not calculate radius: {e}")
        
        # Display results
        self.display_results()
        messagebox.showinfo("Success", "Branch axis and radius calculated! Remember to save the branch set.")
    
    def display_results(self):
        """Display calculation results."""
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', "Branch Calculation Results\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        
        if 'axis_direction' in self.current_results:
            self.results_text.insert(tk.END, f"Axis Direction: {self.current_results['axis_direction']}\n\n")
        
        if 'axis_point' in self.current_results:
            self.results_text.insert(tk.END, f"Axis Point: {self.current_results['axis_point']}\n\n")
        
        if 'radius' in self.current_results:
            self.results_text.insert(tk.END, f"Radius: {self.current_results['radius']:.4f} mm\n\n")
        else:
            self.results_text.insert(tk.END, "Radius: Not calculated (need surface points from at least 2 cameras)\n\n")
    
    def create_new(self):
        """Create a new branch set."""
        self.current_set_name = None
        self.name_var.set('')
        self.cal_var.set('')
        self.current_results = {}
        
        # Clear all text areas
        placeholder = "[0.0, 0.0]\n[0.0, 0.0]"
        for camera in self.camera_list:
            # Clear axis points
            if camera in self.camera_tables:
                text_widget = self.camera_tables[camera]
                text_widget.delete('1.0', tk.END)
                text_widget.insert('1.0', placeholder)
                text_widget.placeholder_set = True
                tree = getattr(self, f'tree_2d_{camera}_axis', None)
                if tree:
                    for item in tree.get_children():
                        tree.delete(item)
            
            # Clear surface points
            if camera in self.surface_tables:
                text_widget = self.surface_tables[camera]
                text_widget.delete('1.0', tk.END)
                text_widget.insert('1.0', placeholder)
                text_widget.placeholder_set = True
                tree = getattr(self, f'tree_2d_{camera}_surface', None)
                if tree:
                    for item in tree.get_children():
                        tree.delete(item)
        
        self.results_text.delete('1.0', tk.END)
        self.name_entry.focus()
    
    def delete_set(self):
        """Delete the selected branch set."""
        selection = self.set_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a branch set to delete.")
            return
        
        set_name = self.set_listbox.get(selection[0])
        if messagebox.askyesno("Confirm", f"Delete branch set '{set_name}'?"):
            del self.branch_sets[set_name]
            save_branch_sets(self.branch_sets)
            self.refresh_list()
            self.create_new()
            messagebox.showinfo("Success", f"Branch set '{set_name}' deleted.")
    
    def load_set(self, set_name):
        """Load a branch set into the editor."""
        if set_name not in self.branch_sets:
            return
        
        self.current_set_name = set_name
        self.name_var.set(set_name)
        branch_set = self.branch_sets[set_name]
        
        # Load calibration set
        cal_name = branch_set.get('calibration_set', '')
        if cal_name in self.calibration_sets:
            self.cal_var.set(cal_name)
            self.on_calibration_selected()
        elif cal_name:
            # Calibration set not found - warn user
            messagebox.showwarning(
                "Missing Calibration Set",
                f"Branch set '{set_name}' references calibration set '{cal_name}', but it was not found.\n\n"
                f"This may happen if the calibration set was deleted or renamed.\n\n"
                f"Please select a different calibration set or recreate '{cal_name}'."
            )
            # Clear the calibration selection
            self.cal_var.set('')
            self.current_calibration_set = None
            self.current_calibration_coefficients = None
        
        # Load branch axis points
        branch_points_2d = branch_set.get('branch_points_2d', {})
        for camera in self.camera_list:
            if camera in branch_points_2d:
                text_widget = self.camera_tables[camera]
                points = branch_points_2d[camera]
                text_lines = [f"[{p[0]}, {p[1]}]" for p in points]
                text_widget.delete('1.0', tk.END)
                text_widget.insert('1.0', '\n'.join(text_lines))
                text_widget.placeholder_set = False
                self.load_2d_from_text(camera, is_axis=True, show_message=False)
        
        # Load surface points
        surface_points_2d = branch_set.get('surface_points_2d', {})
        for camera in self.camera_list:
            if camera in surface_points_2d:
                text_widget = self.surface_tables[camera]
                points = surface_points_2d[camera]
                text_lines = [f"[{p[0]}, {p[1]}]" for p in points]
                text_widget.delete('1.0', tk.END)
                text_widget.insert('1.0', '\n'.join(text_lines))
                text_widget.placeholder_set = False
                self.load_2d_from_text(camera, is_axis=False, show_message=False)
        
        # Load enabled states
        if 'axis_camera_enabled' in branch_set:
            for cam, enabled in branch_set['axis_camera_enabled'].items():
                if cam in self.camera_list:
                    self.axis_camera_enabled[cam] = enabled
                    # Update checkbox if it exists
                    var = getattr(self, f'axis_enable_{cam}', None)
                    if var is not None:
                        var.set(enabled)
        
        if 'surface_camera_enabled' in branch_set:
            for cam, enabled in branch_set['surface_camera_enabled'].items():
                if cam in self.camera_list:
                    self.surface_camera_enabled[cam] = enabled
                    # Update checkbox if it exists
                    var = getattr(self, f'surface_enable_{cam}', None)
                    if var is not None:
                        var.set(enabled)
        
        # Load results
        if 'axis_direction' in branch_set:
            self.current_results['axis_direction'] = branch_set['axis_direction']
        if 'axis_point' in branch_set:
            self.current_results['axis_point'] = branch_set['axis_point']
        if 'radius' in branch_set:
            self.current_results['radius'] = branch_set['radius']
        
        self.display_results()
    
    def save_set(self):
        """Save the current branch set."""
        set_name = self.name_var.get().strip()
        if not set_name:
            messagebox.showerror("Error", "Please enter a branch set name.")
            return
        
        if not self.current_calibration_set:
            messagebox.showerror("Error", "Please select a calibration set.")
            return
        
        # Collect branch axis points
        branch_points_2d = {}
        for camera in self.camera_list:
            points = self.get_2d_points(camera, is_axis=True)
            if points is not None and len(points) > 0:
                branch_points_2d[camera] = points.tolist()
        
        # Collect surface points
        surface_points_2d = {}
        for camera in self.camera_list:
            points = self.get_2d_points(camera, is_axis=False)
            if points is not None and len(points) > 0:
                surface_points_2d[camera] = points.tolist()
        
        # Create branch set data
        branch_set = {
            "name": set_name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "created": datetime.now().isoformat(),
            "calibration_set": self.current_calibration_set,
            "branch_points_2d": branch_points_2d,
            "surface_points_2d": surface_points_2d,
            "cameras_list": self.camera_list.copy(),
            "axis_camera_enabled": {cam: self.axis_camera_enabled.get(cam, True) for cam in self.camera_list},
            "surface_camera_enabled": {cam: self.surface_camera_enabled.get(cam, True) for cam in self.camera_list}
        }
        
        # Include calculated results if available (convert numpy arrays to lists for JSON)
        if 'axis_direction' in self.current_results:
            axis_dir = self.current_results['axis_direction']
            # Convert to list if it's a numpy array
            if isinstance(axis_dir, np.ndarray):
                branch_set['axis_direction'] = axis_dir.tolist()
            else:
                branch_set['axis_direction'] = axis_dir
        if 'axis_point' in self.current_results:
            axis_pt = self.current_results['axis_point']
            # Convert to list if it's a numpy array
            if isinstance(axis_pt, np.ndarray):
                branch_set['axis_point'] = axis_pt.tolist()
            else:
                branch_set['axis_point'] = axis_pt
        if 'radius' in self.current_results:
            branch_set['radius'] = self.current_results['radius']
        
        # Save
        try:
            self.branch_sets[set_name] = branch_set
            save_branch_sets(self.branch_sets)
            self.refresh_list()
            
            # Select the saved set
            items = self.set_listbox.get(0, tk.END)
            if set_name in items:
                idx = list(items).index(set_name)
                self.set_listbox.selection_set(idx)
                self.set_listbox.see(idx)
            
            messagebox.showinfo("Success", f"Branch set '{set_name}' saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save branch set: {e}\n\nPlease check that the file is not open in another program.")
            import traceback
            traceback.print_exc()


def main():
    root = tk.Tk()
    app = BranchSetManager(root)
    root.mainloop()


if __name__ == "__main__":
    main()
