"""
GUI tool to manage calibration sets.
Create, update, delete, and view calibration sets with DLT coefficient calculation.
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
from dlt_utils import calculate_dlt_coefficients

CALIBRATION_SETS_FILE = Path(__file__).parent / "calibration_sets.json"
RAW_DATA_DIR = Path(__file__).parent / "Raw_data"
DEFAULT_CAMERAS = ["Left", "Top", "Right", "Front"]


def load_calibration_sets():
    """Load calibration sets from JSON file."""
    if CALIBRATION_SETS_FILE.exists():
        with open(CALIBRATION_SETS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_calibration_sets(calibration_sets):
    """Save calibration sets to JSON file."""
    with open(CALIBRATION_SETS_FILE, 'w') as f:
        json.dump(calibration_sets, f, indent=2)


class CalibrationSetManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Calibration Set Manager")
        self.root.geometry("1200x800")
        
        self.calibration_sets = load_calibration_sets()
        self.camera_list = DEFAULT_CAMERAS.copy()  # Editable camera list
        self.current_dlt_results = {}  # Store calculated DLT results temporarily
        
        # Main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        # Left panel - List of calibration sets
        left_frame = ttk.LabelFrame(main_frame, text="Calibration Sets", padding="10")
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
        
        # Camera settings button
        ttk.Button(btn_frame, text="Camera Settings", command=self.open_camera_settings).pack(side=tk.LEFT, padx=(10, 0))
        
        # Right panel - Details/Editor
        right_frame = ttk.LabelFrame(main_frame, text="Calibration Set Details", padding="10")
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
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Tab 1: 3D Points
        self.tab_3d = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_3d, text="3D Points")
        self.setup_3d_tab()
        
        # Tab: Results (create before camera tabs so we can reference it)
        self.tab_results = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tab_results, text="Results")
        self.setup_results_tab()
        
        # Camera tabs (created dynamically)
        self.camera_tabs = {}
        self.camera_tables = {}
        self.refresh_camera_tabs()
        
        # Save button
        btn_save_frame = ttk.Frame(right_frame)
        btn_save_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btn_save_frame, text="Save Calibration Set", command=self.save_set).pack(side=tk.LEFT)
        ttk.Button(btn_save_frame, text="Calculate DLT", command=self.calculate_dlt).pack(side=tk.LEFT, padx=(10, 0))
        
        self.refresh_list()
        self.current_set_name = None
    
    def refresh_camera_tabs(self):
        """Refresh camera tabs based on current camera list."""
        # Remove existing camera tabs
        for camera in list(self.camera_tabs.keys()):
            if camera not in self.camera_list:
                tab = self.camera_tabs[camera]
                self.notebook.forget(tab)
                del self.camera_tabs[camera]
                if camera in self.camera_tables:
                    del self.camera_tables[camera]
        
        # Add/update camera tabs
        for camera in self.camera_list:
            if camera not in self.camera_tabs:
                tab = ttk.Frame(self.notebook, padding="10")
                self.notebook.insert(self.notebook.index(self.tab_results), tab, text=f"{camera} Camera")
                self.camera_tabs[camera] = tab
                self.setup_camera_tab(camera, tab)
            else:
                # Update tab name in case camera was renamed
                tab = self.camera_tabs[camera]
                idx = self.notebook.index(tab)
                self.notebook.tab(idx, text=f"{camera} Camera")
    
    def open_camera_settings(self):
        """Open dialog to edit camera list."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Camera Settings")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Instructions
        instructions = ttk.Label(dialog, text="Edit camera names (one per line):", padding="10")
        instructions.pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # Text area for cameras
        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        camera_text = scrolledtext.ScrolledText(text_frame, height=10, width=40)
        camera_text.pack(fill=tk.BOTH, expand=True)
        camera_text.insert('1.0', '\n'.join(self.camera_list))
        
        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def save_cameras():
            text = camera_text.get('1.0', tk.END).strip()
            new_cameras = [line.strip() for line in text.split('\n') if line.strip()]
            if len(new_cameras) == 0:
                messagebox.showerror("Error", "At least one camera is required.")
                return
            if len(new_cameras) != len(set(new_cameras)):
                messagebox.showerror("Error", "Camera names must be unique.")
                return
            
            self.camera_list = new_cameras
            self.refresh_camera_tabs()
            dialog.destroy()
            messagebox.showinfo("Success", f"Camera list updated. Now using {len(self.camera_list)} cameras.")
        
        ttk.Button(btn_frame, text="Save", command=save_cameras).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT)
    
    def setup_3d_tab(self):
        """Setup the 3D points input tab."""
        # Instructions
        instructions = ttk.Label(self.tab_3d, text="Enter 3D calibration points in format: [X, Y, Z] (one per line). Need at least 6 points.", 
                                font=('TkDefaultFont', 9))
        instructions.pack(anchor=tk.W, pady=(0, 10))
        
        # Text input area
        text_frame = ttk.Frame(self.tab_3d)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.text_3d = scrolledtext.ScrolledText(text_frame, height=12, width=40, font=('Courier', 10))
        self.text_3d.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder text
        placeholder = "[0, 0, 0]\n[0, 0, -2]\n[0, 2, -2]\n[0, 2, 0]\n[2, 0, -2]\n[2, 2, -2]\n[2, 0, 0]\n[2, 2, 0]"
        self.text_3d.insert('1.0', placeholder)
        self.text_3d.bind('<FocusIn>', self.on_3d_focus_in)
        
        # Button frame
        btn_frame = ttk.Frame(self.tab_3d)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Load from Text", command=self.load_3d_from_text).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Clear", command=self.clear_3d_text).pack(side=tk.LEFT, padx=(0, 5))
        
        # Table frame (for viewing parsed points)
        table_label = ttk.Label(self.tab_3d, text="Parsed Points:", font=('TkDefaultFont', 9, 'bold'))
        table_label.pack(anchor=tk.W, pady=(10, 5))
        
        table_frame = ttk.Frame(self.tab_3d)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for 3D points
        columns = ('Point', 'X', 'Y', 'Z')
        self.tree_3d = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.tree_3d.heading(col, text=col)
            self.tree_3d.column(col, width=100)
        
        # Scrollbar
        scrollbar_3d = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree_3d.yview)
        self.tree_3d.configure(yscrollcommand=scrollbar_3d.set)
        
        self.tree_3d.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_3d.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_camera_tab(self, camera, tab):
        """Setup a camera's 2D points input tab."""
        # Instructions with emphasis on camera name
        instructions = ttk.Label(tab, 
                                text=f"⚠ IMPORTANT: Enter 2D points for the {camera} camera ONLY.\n"
                                     f"Format: [u, v] (one per line). Points must match 3D points order.", 
                                font=('TkDefaultFont', 9),
                                foreground='darkblue')
        instructions.pack(anchor=tk.W, pady=(0, 10))
        
        # Text input area
        text_frame = ttk.Frame(tab)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        text_widget = scrolledtext.ScrolledText(text_frame, height=12, width=40, font=('Courier', 10))
        text_widget.pack(fill=tk.BOTH, expand=True)
        self.camera_tables[camera] = text_widget
        
        # Placeholder text
        placeholder = "[0.0, 0.0]\n[0.0, 0.0]\n[0.0, 0.0]\n[0.0, 0.0]\n[0.0, 0.0]\n[0.0, 0.0]\n[0.0, 0.0]\n[0.0, 0.0]"
        text_widget.insert('1.0', placeholder)
        text_widget.placeholder_set = True
        
        def on_focus_in(event):
            if text_widget.placeholder_set:
                text_widget.delete('1.0', tk.END)
                text_widget.placeholder_set = False
        
        text_widget.bind('<FocusIn>', on_focus_in)
        
        # Button frame
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X)
        
        def load_from_text():
            self.load_2d_from_text(camera)
        
        def clear_text():
            text_widget.delete('1.0', tk.END)
            text_widget.insert('1.0', placeholder)
            text_widget.placeholder_set = True
            # Clear table
            if hasattr(self, 'tree_2d_' + camera):
                tree = getattr(self, 'tree_2d_' + camera)
                for item in tree.get_children():
                    tree.delete(item)
        
        def save_points():
            self.save_camera_points(camera)
        
        def load_points():
            self.load_camera_points(camera)
        
        ttk.Button(btn_frame, text="Load from Text", command=load_from_text).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text=f"Save {camera} Points", command=save_points).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text=f"Load {camera} Points", command=load_points).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Clear", command=clear_text).pack(side=tk.LEFT, padx=(0, 5))
        
        # Table frame (for viewing parsed points)
        table_label = ttk.Label(tab, text="Parsed Points:", font=('TkDefaultFont', 9, 'bold'))
        table_label.pack(anchor=tk.W, pady=(10, 5))
        
        table_frame = ttk.Frame(tab)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for 2D points
        columns = ('Point', 'u', 'v')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        setattr(self, 'tree_2d_' + camera, tree)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_results_tab(self):
        """Setup the results display tab."""
        # Results text area
        self.results_text = scrolledtext.ScrolledText(self.tab_results, height=20, width=60)
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def refresh_list(self):
        """Refresh the list of calibration sets."""
        self.set_listbox.delete(0, tk.END)
        for name in sorted(self.calibration_sets.keys()):
            self.set_listbox.insert(tk.END, name)
    
    def on_set_select(self, event):
        """Handle selection of a calibration set."""
        selection = self.set_listbox.curselection()
        if not selection:
            return
        
        set_name = self.set_listbox.get(selection[0])
        self.load_set(set_name)
    
    def load_set(self, set_name):
        """Load a calibration set into the editor."""
        if set_name not in self.calibration_sets:
            return
        
        self.current_set_name = set_name
        self.name_var.set(set_name)
        cal_set = self.calibration_sets[set_name]
        
        # Load camera list from saved set if available, otherwise use current
        if 'cameras_list' in cal_set:
            saved_cameras = cal_set['cameras_list']
            if saved_cameras != self.camera_list:
                # Update camera list and refresh tabs
                self.camera_list = saved_cameras
                self.refresh_camera_tabs()
        
        # Verify that all cameras in the saved set have data
        saved_camera_names = set(cal_set.get('cameras', {}).keys())
        current_camera_names = set(self.camera_list)
        if saved_camera_names != current_camera_names:
            missing = current_camera_names - saved_camera_names
            extra = saved_camera_names - current_camera_names
            if missing:
                messagebox.showwarning("Warning", 
                    f"Calibration set '{set_name}' is missing data for cameras: {', '.join(missing)}.\n"
                    f"This may cause issues when using this calibration set.")
            if extra:
                messagebox.showinfo("Info", 
                    f"Calibration set '{set_name}' has data for cameras not in current list: {', '.join(extra)}.")
        
        # Load 3D points
        self.text_3d.delete('1.0', tk.END)
        for item in self.tree_3d.get_children():
            self.tree_3d.delete(item)
        
        if 'points_3d' in cal_set:
            # Format as text
            text_lines = [f"[{p[0]}, {p[1]}, {p[2]}]" for p in cal_set['points_3d']]
            self.text_3d.insert('1.0', '\n'.join(text_lines))
            # Also add to table
            for i, point in enumerate(cal_set['points_3d'], 1):
                self.tree_3d.insert('', tk.END, values=(i, point[0], point[1], point[2]))
        
        # Load 2D points for each camera
        for camera in self.camera_list:
            text_widget = self.camera_tables[camera]
            tree = getattr(self, 'tree_2d_' + camera, None)
            
            # Clear text and table
            text_widget.delete('1.0', tk.END)
            if tree:
                for item in tree.get_children():
                    tree.delete(item)
            
            if 'cameras' in cal_set and camera in cal_set['cameras']:
                cam_data = cal_set['cameras'][camera]
                if 'points_2d' in cam_data:
                    # Format as text
                    text_lines = [f"[{p[0]}, {p[1]}]" for p in cam_data['points_2d']]
                    text_widget.insert('1.0', '\n'.join(text_lines))
                    text_widget.placeholder_set = False
                    
                    # Also add to table
                    if tree:
                        for i, point in enumerate(cam_data['points_2d'], 1):
                            tree.insert('', tk.END, values=(i, point[0], point[1]))
        
        # Load results if available
        self.display_results(cal_set)
    
    def on_3d_focus_in(self, event):
        """Clear placeholder when 3D text area gets focus."""
        content = self.text_3d.get('1.0', tk.END).strip()
        placeholder = "[0, 0, 0]\n[0, 0, -2]\n[0, 2, -2]\n[0, 2, 0]\n[2, 0, -2]\n[2, 2, -2]\n[2, 0, 0]\n[2, 2, 0]"
        if content == placeholder:
            self.text_3d.delete('1.0', tk.END)
    
    def parse_3d_text(self, text):
        """Parse 3D points from text format [X, Y, Z]."""
        points = []
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try to parse [x, y, z] format
            try:
                # Remove brackets and split by comma
                line = line.strip('[]')
                coords = [float(x.strip()) for x in line.split(',')]
                if len(coords) == 3:
                    points.append(coords)
            except (ValueError, AttributeError):
                continue
        return np.array(points) if points else None
    
    def parse_2d_text(self, text):
        """Parse 2D points from text format [u, v]."""
        points = []
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try to parse [u, v] format
            try:
                # Remove brackets and split by comma
                line = line.strip('[]')
                coords = [float(x.strip()) for x in line.split(',')]
                if len(coords) == 2:
                    points.append(coords)
            except (ValueError, AttributeError):
                continue
        return np.array(points) if points else None
    
    def load_3d_from_text(self):
        """Load 3D points from text area into table."""
        text = self.text_3d.get('1.0', tk.END)
        points = self.parse_3d_text(text)
        
        if points is None or len(points) == 0:
            messagebox.showerror("Error", "Could not parse 3D points. Use format: [X, Y, Z] (one per line)")
            return
        
        # Clear table
        for item in self.tree_3d.get_children():
            self.tree_3d.delete(item)
        
        # Add points to table
        for i, point in enumerate(points, 1):
            self.tree_3d.insert('', tk.END, values=(i, point[0], point[1], point[2]))
        
        messagebox.showinfo("Success", f"Loaded {len(points)} 3D points")
    
    def load_2d_from_text(self, camera):
        """Load 2D points from text area into table for a camera."""
        text_widget = self.camera_tables[camera]
        text = text_widget.get('1.0', tk.END)
        points = self.parse_2d_text(text)
        
        if points is None or len(points) == 0:
            messagebox.showerror("Error", f"Could not parse 2D points for {camera} camera. Use format: [u, v] (one per line)")
            return
        
        # Get table
        tree = getattr(self, 'tree_2d_' + camera)
        
        # Clear table
        for item in tree.get_children():
            tree.delete(item)
        
        # Add points to table
        for i, point in enumerate(points, 1):
            tree.insert('', tk.END, values=(i, point[0], point[1]))
        
        messagebox.showinfo("Success", f"Loaded {len(points)} 2D points for {camera} camera")
    
    def clear_3d_text(self):
        """Clear 3D text area and reset placeholder."""
        placeholder = "[0, 0, 0]\n[0, 0, -2]\n[0, 2, -2]\n[0, 2, 0]\n[2, 0, -2]\n[2, 2, -2]\n[2, 0, 0]\n[2, 2, 0]"
        self.text_3d.delete('1.0', tk.END)
        self.text_3d.insert('1.0', placeholder)
        # Clear table
        for item in self.tree_3d.get_children():
            self.tree_3d.delete(item)
    
    def create_new(self):
        """Create a new calibration set."""
        self.current_set_name = None
        self.name_var.set('')
        self.current_dlt_results = {}  # Clear calculated results
        
        # Clear 3D text and table
        placeholder_3d = "[0, 0, 0]\n[0, 0, -2]\n[0, 2, -2]\n[0, 2, 0]\n[2, 0, -2]\n[2, 2, -2]\n[2, 0, 0]\n[2, 2, 0]"
        self.text_3d.delete('1.0', tk.END)
        self.text_3d.insert('1.0', placeholder_3d)
        for item in self.tree_3d.get_children():
            self.tree_3d.delete(item)
        
        # Clear 2D text and tables
        placeholder_2d = "[0.0, 0.0]\n[0.0, 0.0]\n[0.0, 0.0]\n[0.0, 0.0]\n[0.0, 0.0]\n[0.0, 0.0]\n[0.0, 0.0]\n[0.0, 0.0]"
        for camera in self.camera_list:
            text_widget = self.camera_tables[camera]
            text_widget.delete('1.0', tk.END)
            text_widget.insert('1.0', placeholder_2d)
            text_widget.placeholder_set = True
            tree = getattr(self, 'tree_2d_' + camera, None)
            if tree:
                for item in tree.get_children():
                    tree.delete(item)
        
        self.results_text.delete('1.0', tk.END)
        self.name_entry.focus()
    
    def delete_set(self):
        """Delete the selected calibration set."""
        selection = self.set_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a calibration set to delete.")
            return
        
        set_name = self.set_listbox.get(selection[0])
        
        # Check if any branch sets reference this calibration set
        branch_sets_file = Path(__file__).parent.parent / "Branch_Sets" / "branch_sets.json"
        dependent_branch_sets = []
        if branch_sets_file.exists():
            try:
                with open(branch_sets_file, 'r') as f:
                    branch_sets = json.load(f)
                for branch_name, branch_data in branch_sets.items():
                    if branch_data.get('calibration_set') == set_name:
                        dependent_branch_sets.append(branch_name)
            except Exception:
                pass  # If we can't read branch sets, continue anyway
        
        # Warn about dependent branch sets
        if dependent_branch_sets:
            warning_msg = (
                f"WARNING: The following branch sets reference calibration set '{set_name}':\n\n"
                f"{', '.join(dependent_branch_sets)}\n\n"
                f"If you delete this calibration set, these branch sets will no longer be able to "
                f"calculate branch data until you assign them a different calibration set.\n\n"
                f"Are you sure you want to delete '{set_name}'?"
            )
            if not messagebox.askyesno("Confirm Deletion", warning_msg):
                return
        else:
            if not messagebox.askyesno("Confirm", f"Delete calibration set '{set_name}'?"):
                return
        
        del self.calibration_sets[set_name]
        save_calibration_sets(self.calibration_sets)
        self.refresh_list()
        self.create_new()
        messagebox.showinfo("Success", f"Calibration set '{set_name}' deleted.")
    
    def get_3d_points(self):
        """Get 3D points from the table."""
        points = []
        for item in self.tree_3d.get_children():
            values = self.tree_3d.item(item, 'values')
            points.append([float(values[1]), float(values[2]), float(values[3])])
        return np.array(points) if points else None
    
    def get_2d_points(self, camera):
        """Get 2D points for a camera from the table."""
        tree = getattr(self, 'tree_2d_' + camera, None)
        if tree is None:
            return None
        points = []
        for item in tree.get_children():
            values = tree.item(item, 'values')
            points.append([float(values[1]), float(values[2])])
        return np.array(points) if points else None
    
    def calculate_dlt(self):
        """Calculate DLT coefficients for all cameras."""
        # First load 3D points from text
        self.load_3d_from_text()
        points_3d = self.get_3d_points()
        if points_3d is None or len(points_3d) < 6:
            messagebox.showerror("Error", "Need at least 6 3D points to calculate DLT coefficients.")
            return
        
        results = {}
        all_valid = True
        
        for camera in self.camera_list:
            # Parse 2D points from text (silently)
            text_widget = self.camera_tables[camera]
            text = text_widget.get('1.0', tk.END)
            points_2d = self.parse_2d_text(text)
            
            if points_2d is None or len(points_2d) < 6:
                results[camera] = {"error": "Not enough 2D points (need at least 6)"}
                all_valid = False
                continue
            
            if len(points_2d) != len(points_3d):
                results[camera] = {"error": f"Number of 2D points ({len(points_2d)}) doesn't match 3D points ({len(points_3d)})"}
                all_valid = False
                continue
            
            # Update table
            tree = getattr(self, 'tree_2d_' + camera, None)
            if tree:
                for item in tree.get_children():
                    tree.delete(item)
                for i, point in enumerate(points_2d, 1):
                    tree.insert('', tk.END, values=(i, point[0], point[1]))
            
            try:
                # Filter out NaN points
                valid_mask = ~np.isnan(points_2d).any(axis=1)
                valid_3d = points_3d[valid_mask]
                valid_2d = points_2d[valid_mask]
                
                if len(valid_3d) < 6:
                    results[camera] = {"error": "Not enough valid points after filtering"}
                    all_valid = False
                    continue
                
                coefficients, residual = calculate_dlt_coefficients(valid_3d, valid_2d)
                results[camera] = {
                    "coefficients": coefficients.tolist(),
                    "residual": float(residual),
                    "n_points": len(valid_3d)
                }
                # Store results for saving
                self.current_dlt_results[camera] = {
                    "coefficients": coefficients.tolist(),
                    "residual": float(residual)
                }
            except Exception as e:
                results[camera] = {"error": str(e)}
                all_valid = False
        
        # Display results
        self.display_calculation_results(results)
        
        if all_valid:
            messagebox.showinfo("Success", "DLT coefficients calculated successfully! Remember to save the calibration set to store them.")
        else:
            messagebox.showwarning("Warning", "Some cameras had errors. Check the Results tab.")
    
    def display_calculation_results(self, results):
        """Display calculation results in the results tab."""
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert('1.0', "DLT Calculation Results\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        
        for camera in self.camera_list:
            self.results_text.insert(tk.END, f"{camera} Camera:\n")
            if "error" in results.get(camera, {}):
                self.results_text.insert(tk.END, f"  Error: {results[camera]['error']}\n\n")
            else:
                cam_result = results[camera]
                self.results_text.insert(tk.END, f"  Residual: {cam_result['residual']:.4f} pixels\n")
                self.results_text.insert(tk.END, f"  Points used: {cam_result['n_points']}\n")
                self.results_text.insert(tk.END, f"  Coefficients: {cam_result['coefficients']}\n\n")
    
    def display_results(self, cal_set):
        """Display stored results for a loaded set."""
        self.results_text.delete('1.0', tk.END)
        if 'cameras' not in cal_set:
            self.results_text.insert('1.0', "No results available. Calculate DLT coefficients first.")
            return
        
        self.results_text.insert('1.0', f"Calibration Set: {cal_set.get('name', 'Unknown')}\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        
        for camera in self.camera_list:
            if camera in cal_set.get('cameras', {}):
                cam_data = cal_set['cameras'][camera]
                self.results_text.insert(tk.END, f"{camera} Camera:\n")
                if 'residual' in cam_data:
                    self.results_text.insert(tk.END, f"  Residual: {cam_data['residual']:.4f} pixels\n")
                if 'coefficients' in cam_data:
                    self.results_text.insert(tk.END, f"  Coefficients: {cam_data['coefficients']}\n")
                self.results_text.insert(tk.END, "\n")
    
    def save_set(self):
        """Save the current calibration set."""
        set_name = self.name_var.get().strip()
        if not set_name:
            messagebox.showerror("Error", "Please enter a calibration set name.")
            return
        
        # Parse 3D points from text
        text = self.text_3d.get('1.0', tk.END)
        points_3d = self.parse_3d_text(text)
        if points_3d is None or len(points_3d) < 6:
            messagebox.showerror("Error", "Need at least 6 3D points.")
            return
        
        # Collect 2D points for each camera (parse from text)
        cameras_data = {}
        for camera in self.camera_list:
            text_widget = self.camera_tables[camera]
            text = text_widget.get('1.0', tk.END)
            points_2d = self.parse_2d_text(text)
            if points_2d is not None and len(points_2d) > 0:
                cameras_data[camera] = {
                    "points_2d": points_2d.tolist()
                }
        
        # Create calibration set data
        cal_set = {
            "name": set_name,
            "points_3d": points_3d.tolist(),
            "cameras": cameras_data,
            "cameras_list": self.camera_list.copy(),  # Save camera list
            "created": datetime.now().isoformat()
        }
        
        # Include calculated DLT coefficients if they were just calculated
        for camera in self.camera_list:
            if camera in self.current_dlt_results:
                if camera not in cal_set['cameras']:
                    cal_set['cameras'][camera] = {}
                cal_set['cameras'][camera]['coefficients'] = self.current_dlt_results[camera]['coefficients']
                cal_set['cameras'][camera]['residual'] = self.current_dlt_results[camera]['residual']
        
        # If this is an existing set with calculated coefficients, preserve them (if not overwritten above)
        if set_name in self.calibration_sets and 'cameras' in self.calibration_sets[set_name]:
            for camera in self.camera_list:
                if camera in self.calibration_sets[set_name]['cameras']:
                    old_cam = self.calibration_sets[set_name]['cameras'][camera]
                    # Only preserve if we didn't just calculate new ones
                    if camera not in self.current_dlt_results:
                        if 'coefficients' in old_cam:
                            if camera not in cal_set['cameras']:
                                cal_set['cameras'][camera] = {}
                            cal_set['cameras'][camera]['coefficients'] = old_cam['coefficients']
                            if 'residual' in old_cam:
                                cal_set['cameras'][camera]['residual'] = old_cam['residual']
        
        # Save
        self.calibration_sets[set_name] = cal_set
        save_calibration_sets(self.calibration_sets)
        self.refresh_list()
        
        # Select the saved set
        items = self.set_listbox.get(0, tk.END)
        if set_name in items:
            idx = list(items).index(set_name)
            self.set_listbox.selection_set(idx)
            self.set_listbox.see(idx)
        
        messagebox.showinfo("Success", f"Calibration set '{set_name}' saved successfully!")
    
    def save_3d_points(self):
        """Save 3D points to a file."""
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save 3D Points"
        )
        if filename:
            text = self.text_3d.get('1.0', tk.END)
            with open(filename, 'w') as f:
                f.write(text)
            messagebox.showinfo("Success", f"3D points saved to {filename}")
    
    def load_3d_points(self):
        """Load 3D points from a file."""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Load 3D Points"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    text = f.read()
                self.text_3d.delete('1.0', tk.END)
                self.text_3d.insert('1.0', text)
                self.load_3d_from_text()
                messagebox.showinfo("Success", f"3D points loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")
    
    def save_camera_points(self, camera):
        """Save 2D points for a camera to a file."""
        from tkinter import filedialog
        
        # Check if camera exists in camera_tables
        if camera not in self.camera_tables:
            messagebox.showerror("Error", f"Camera '{camera}' not found. Please ensure the camera is in the camera list.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title=f"Save {camera} Camera Points"
        )
        if filename:
            try:
                text_widget = self.camera_tables[camera]
                text = text_widget.get('1.0', tk.END)
                with open(filename, 'w') as f:
                    f.write(text)
                messagebox.showinfo("Success", f"{camera} camera points saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")
    
    def load_camera_points(self, camera):
        """Load 2D points for a camera from a file."""
        from tkinter import filedialog
        
        # Check if camera exists in camera_tables
        if camera not in self.camera_tables:
            messagebox.showerror("Error", f"Camera '{camera}' not found. Please ensure the camera is in the camera list.")
            return
        
        filename = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title=f"Load {camera} Camera Points"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    text = f.read()
                text_widget = self.camera_tables[camera]
                text_widget.delete('1.0', tk.END)
                text_widget.insert('1.0', text)
                text_widget.placeholder_set = False
                self.load_2d_from_text(camera)
                messagebox.showinfo("Success", f"{camera} camera points loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")


def main():
    root = tk.Tk()
    app = CalibrationSetManager(root)
    root.mainloop()


if __name__ == "__main__":
    main()
