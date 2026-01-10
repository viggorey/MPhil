"""
GUI for 3D Reconstruction and Parameterization.
Allows selecting datasets, viewing status, and running operations.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import threading
from pathlib import Path
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
# Force a GUI backend - must be done BEFORE importing pyplot
# Try TkAgg first (compatible with Tkinter), then Qt5Agg, then QtAgg
gui_backends = ['TkAgg', 'Qt5Agg', 'QtAgg']
backend_set = False
for backend in gui_backends:
    try:
        matplotlib.use(backend, force=True)
        backend_set = True
        print(f"DEBUG: Set matplotlib backend to {backend}")
        break
    except Exception as e:
        print(f"DEBUG: Could not set {backend} backend: {e}")
        continue

if not backend_set:
    print("WARNING: Could not set any GUI backend, using default (may not show windows)")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import CheckButtons

# Verify backend is set correctly after import
if matplotlib.get_backend() == 'Agg' or 'Agg' in matplotlib.get_backend():
    print(f"WARNING: Backend is still Agg after import! Attempting to fix...")
    try:
        matplotlib.use('TkAgg', force=True)
        import importlib
        importlib.reload(plt)
        print(f"Fixed: Backend is now {matplotlib.get_backend()}")
    except:
        try:
            matplotlib.use('Qt5Agg', force=True)
            import importlib
            importlib.reload(plt)
            print(f"Fixed: Backend is now {matplotlib.get_backend()}")
        except:
            print(f"ERROR: Could not fix backend, still {matplotlib.get_backend()}")

# Add paths for imports
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "Data"
CONFIG_DIR = BASE_DIR / "Config"
INPUT_DIR = Path(__file__).parent.parent / "Data" / "Datasets" / "3D_data"
OUTPUT_DIR = Path(__file__).parent.parent / "Data" / "Datasets" / "3D_data_params"
RECONSTRUCT_OUTPUT_DIR = Path(__file__).parent.parent / "Data" / "Datasets" / "3D_data"

# Import the processing functions
sys.path.insert(0, str(Path(__file__).parent))
from master_reconstruct import load_dataset_links, reconstruct_dataset, load_branch_set
from master_parameterize import (load_3d_data, detect_species, load_species_data, calculate_ant_size_normalization,
                                 calculate_all_parameters, save_parameterized_data, load_config, 
                                 load_dataset_links as load_links_param, trim_data, FOOT_POINTS)
from parameterize_utils import check_foot_attachment


def load_dataset_links_for_reconstruct():
    """Load dataset links for reconstruction."""
    links_file = DATA_DIR / "dataset_links.json"
    if links_file.exists():
        with open(links_file, 'r') as f:
            return json.load(f)
    return {}


def get_reconstructed_datasets():
    """Get list of datasets that have been 3D reconstructed."""
    if not RECONSTRUCT_OUTPUT_DIR.exists():
        return set()
    return {f.stem for f in RECONSTRUCT_OUTPUT_DIR.glob("*.xlsx")}


def get_parameterized_datasets():
    """Get list of datasets that have been parameterized."""
    if not OUTPUT_DIR.exists():
        return set()
    return {f.stem.replace('_param', '') for f in OUTPUT_DIR.glob("*_param.xlsx")}


def get_available_3d_datasets():
    """Get list of available 3D datasets."""
    if not INPUT_DIR.exists():
        return []
    return sorted([f.stem for f in INPUT_DIR.glob("*.xlsx")])


class ProcessingGUI:
    def __init__(self, root, embedded_mode=False, default_tab=0):
        self.root = root
        self.embedded_mode = embedded_mode
        
        if not embedded_mode:
            self.root.title("3D Processing Manager")
            self.root.geometry("1200x800")
        
        self.reconstructed = get_reconstructed_datasets()
        self.parameterized = get_parameterized_datasets()
        self.available_3d = get_available_3d_datasets()
        self.dataset_links = load_dataset_links_for_reconstruct()
        
        self.is_processing = False
        self.current_operation = None
        
        # Main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        if embedded_mode:
            # In embedded mode, just create the requested tab directly without notebook
            if default_tab == 0:
                self.tab_reconstruct = ttk.Frame(main_frame, padding="10")
                self.tab_reconstruct.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
                main_frame.columnconfigure(0, weight=1)
                main_frame.rowconfigure(0, weight=1)
                self.setup_reconstruction_tab()
                self.notebook = None  # No notebook in embedded mode
            else:
                self.tab_parameterize = ttk.Frame(main_frame, padding="10")
                self.tab_parameterize.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
                main_frame.columnconfigure(0, weight=1)
                main_frame.rowconfigure(0, weight=1)
                self.setup_parameterization_tab()
                self.notebook = None  # No notebook in embedded mode
        else:
            # Notebook for tabs (only when not embedded)
            self.notebook = ttk.Notebook(main_frame)
            self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
            main_frame.columnconfigure(0, weight=1)
            main_frame.rowconfigure(0, weight=1)
            
            # Tab 1: Reconstruction
            self.tab_reconstruct = ttk.Frame(self.notebook, padding="10")
            self.notebook.add(self.tab_reconstruct, text="3D Reconstruction")
            self.setup_reconstruction_tab()
            
            # Tab 2: Parameterization
            self.tab_parameterize = ttk.Frame(self.notebook, padding="10")
            self.notebook.add(self.tab_parameterize, text="Parameterization")
            self.setup_parameterization_tab()
        
        # Status bar (only show when not embedded)
        if not embedded_mode:
            self.status_var = tk.StringVar(value="Ready")
            status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
            status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        else:
            self.status_var = tk.StringVar(value="Ready")
        
        # Initialize
        if default_tab == 0 or not embedded_mode:
            self.refresh_reconstruction_list()
        if default_tab == 1 or not embedded_mode:
            self.refresh_parameterization_list()
    
    def setup_reconstruction_tab(self):
        """Setup the 3D Reconstruction tab."""
        # Left panel - Dataset list
        left_frame = ttk.LabelFrame(self.tab_reconstruct, text="Linked Datasets", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        self.tab_reconstruct.columnconfigure(0, weight=1)
        self.tab_reconstruct.rowconfigure(0, weight=1)
        
        # Select All checkbox
        select_all_frame = ttk.Frame(left_frame)
        select_all_frame.pack(fill=tk.X, pady=(0, 10))
        self.reconstruct_select_all_var = tk.BooleanVar()
        ttk.Checkbutton(select_all_frame, text="Select All", 
                       variable=self.reconstruct_select_all_var,
                       command=self.toggle_reconstruct_all).pack(side=tk.LEFT)
        ttk.Button(select_all_frame, text="Refresh", command=self.refresh_reconstruction_list).pack(side=tk.RIGHT)
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview for better display
        columns = ('Dataset', 'Calibration', 'Branch', 'Species', 'Status')
        self.reconstruct_tree = ttk.Treeview(list_frame, columns=columns, show='tree headings', 
                                            yscrollcommand=scrollbar.set, height=20, selectmode='none')
        self.reconstruct_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.reconstruct_tree.yview)
        
        # Configure columns
        self.reconstruct_tree.heading('#0', text='')
        self.reconstruct_tree.column('#0', width=30, stretch=False)
        for col in columns:
            self.reconstruct_tree.heading(col, text=col)
            if col == 'Dataset':
                self.reconstruct_tree.column(col, width=100)
            elif col == 'Status':
                self.reconstruct_tree.column(col, width=150)
            else:
                self.reconstruct_tree.column(col, width=120)
        
        # Right panel - Controls and log
        right_frame = ttk.LabelFrame(self.tab_reconstruct, text="Controls", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.tab_reconstruct.columnconfigure(1, weight=1)
        
        # Run button
        self.reconstruct_btn = ttk.Button(right_frame, text="Run Reconstruction", 
                                         command=self.run_reconstruction)
        self.reconstruct_btn.pack(pady=(0, 10))
        
        # Progress bar
        self.reconstruct_progress = ttk.Progressbar(right_frame, mode='determinate')
        self.reconstruct_progress.pack(fill=tk.X, pady=(0, 10))
        
        # Log area
        log_label = ttk.Label(right_frame, text="Log:")
        log_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.reconstruct_log = scrolledtext.ScrolledText(right_frame, height=25, width=50, wrap=tk.WORD)
        self.reconstruct_log.pack(fill=tk.BOTH, expand=True)
    
    def setup_parameterization_tab(self):
        """Setup the Parameterization tab."""
        # Left panel - Dataset list
        left_frame = ttk.LabelFrame(self.tab_parameterize, text="3D Datasets", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        self.tab_parameterize.columnconfigure(0, weight=1)
        self.tab_parameterize.rowconfigure(0, weight=1)
        
        # Select All checkbox
        select_all_frame = ttk.Frame(left_frame)
        select_all_frame.pack(fill=tk.X, pady=(0, 10))
        self.param_select_all_var = tk.BooleanVar()
        ttk.Checkbutton(select_all_frame, text="Select All", 
                       variable=self.param_select_all_var,
                       command=self.toggle_parameterize_all).pack(side=tk.LEFT)
        ttk.Button(select_all_frame, text="Refresh", command=self.refresh_parameterization_list).pack(side=tk.RIGHT)
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview
        columns = ('Dataset', 'Species', 'Frames', 'Status')
        self.param_tree = ttk.Treeview(list_frame, columns=columns, show='tree headings',
                                      yscrollcommand=scrollbar.set, height=20, selectmode='none')
        self.param_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.param_tree.yview)
        
        # Configure columns
        self.param_tree.heading('#0', text='')
        self.param_tree.column('#0', width=30, stretch=False)
        for col in columns:
            self.param_tree.heading(col, text=col)
            if col == 'Dataset':
                self.param_tree.column(col, width=120)
            elif col == 'Status':
                self.param_tree.column(col, width=150)
            else:
                self.param_tree.column(col, width=100)
        
        # Right panel - Controls and log
        right_frame = ttk.LabelFrame(self.tab_parameterize, text="Controls", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.tab_parameterize.columnconfigure(1, weight=1)
        
        # Trim options
        trim_frame = ttk.LabelFrame(right_frame, text="Trimming Options", padding="10")
        trim_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.trim_enable_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(trim_frame, text="Enable trimming", 
                      variable=self.trim_enable_var).pack(anchor=tk.W)
        
        condition_frame = ttk.Frame(trim_frame)
        condition_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(condition_frame, text="Condition:").pack(side=tk.LEFT, padx=(0, 10))
        self.trim_condition_var = tk.StringVar(value='gait_cycle')
        ttk.Combobox(condition_frame, textvariable=self.trim_condition_var,
                    values=['gait_cycle', 'none'], state='readonly', width=15).pack(side=tk.LEFT)
        
        # Buttons frame
        buttons_frame = ttk.Frame(right_frame)
        buttons_frame.pack(pady=(0, 10))
        
        # Run Parameterization button
        self.param_btn = ttk.Button(buttons_frame, text="Run Parameterization",
                                   command=self.run_parameterization)
        self.param_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # View Interactive Figure button
        self.view_fig_btn = ttk.Button(buttons_frame, text="View Interactive Figure",
                                      command=self.view_interactive_figure)
        self.view_fig_btn.pack(side=tk.LEFT)
        
        # Progress bar
        self.param_progress = ttk.Progressbar(right_frame, mode='determinate')
        self.param_progress.pack(fill=tk.X, pady=(0, 10))
        
        # Log area
        log_label = ttk.Label(right_frame, text="Log:")
        log_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.param_log = scrolledtext.ScrolledText(right_frame, height=20, width=50, wrap=tk.WORD)
        self.param_log.pack(fill=tk.BOTH, expand=True)
    
    def refresh_reconstruction_list(self):
        """Refresh the reconstruction dataset list."""
        # Clear existing items
        for item in self.reconstruct_tree.get_children():
            self.reconstruct_tree.delete(item)
        
        # Update status
        self.reconstructed = get_reconstructed_datasets()
        self.dataset_links = load_dataset_links_for_reconstruct()
        
        # Add datasets
        for dataset_name, link in sorted(self.dataset_links.items()):
            cal = link.get('calibration_set', 'N/A')
            branch = link.get('branch_set', 'N/A')
            species = link.get('species', 'N/A')
            
            # Determine status
            if dataset_name in self.reconstructed:
                status = "[OK] Reconstructed"
                status_color = 'green'
            else:
                status = "Not reconstructed"
                status_color = 'gray'
            
            item = self.reconstruct_tree.insert('', tk.END, text='☐', 
                                               values=(dataset_name, cal, branch, species, status),
                                               tags=(status_color,))
        
        # Configure tags for colors
        self.reconstruct_tree.tag_configure('green', foreground='green')
        self.reconstruct_tree.tag_configure('gray', foreground='gray')
        self.reconstruct_tree.tag_configure('red', foreground='red')
        
        # Bind click events for checkboxes
        self.reconstruct_tree.bind('<Button-1>', self.on_reconstruct_tree_click)
    
    def refresh_parameterization_list(self):
        """Refresh the parameterization dataset list."""
        # Clear existing items
        for item in self.param_tree.get_children():
            self.param_tree.delete(item)
        
        # Update status
        self.parameterized = get_parameterized_datasets()
        self.available_3d = get_available_3d_datasets()
        dataset_links = load_links_param()
        
        # Add datasets
        for dataset_name in self.available_3d:
            # Get species from links
            species = 'Unknown'
            if dataset_name in dataset_links:
                species = dataset_links[dataset_name].get('species', 'Unknown')
            
            # Get frame count
            try:
                data = load_3d_data(dataset_name)
                frames = data['frames']
            except:
                frames = 'N/A'
            
            # Determine status
            if dataset_name in self.parameterized:
                status = "[OK] Parameterized"
                status_color = 'green'
            else:
                status = "Not parameterized"
                status_color = 'gray'
            
            item = self.param_tree.insert('', tk.END, text='☐',
                                          values=(dataset_name, species, frames, status),
                                          tags=(status_color,))
        
        # Configure tags
        self.param_tree.tag_configure('green', foreground='green')
        self.param_tree.tag_configure('gray', foreground='gray')
        self.param_tree.tag_configure('red', foreground='red')
        self.param_tree.tag_configure('yellow', foreground='orange')
        
        # Bind click events for checkboxes
        self.param_tree.bind('<Button-1>', self.on_param_tree_click)
    
    def toggle_reconstruct_all(self):
        """Toggle select all for reconstruction."""
        select_all = self.reconstruct_select_all_var.get()
        for item in self.reconstruct_tree.get_children():
            self.reconstruct_tree.item(item, text='☑' if select_all else '☐')
    
    def toggle_parameterize_all(self):
        """Toggle select all for parameterization."""
        select_all = self.param_select_all_var.get()
        for item in self.param_tree.get_children():
            self.param_tree.item(item, text='☑' if select_all else '☐')
    
    def get_selected_reconstruct_datasets(self):
        """Get list of selected datasets for reconstruction."""
        selected = []
        for item in self.reconstruct_tree.get_children():
            checkbox_text = self.reconstruct_tree.item(item, 'text')
            if checkbox_text == '☑':
                dataset_name = self.reconstruct_tree.item(item, 'values')[0]
                selected.append(dataset_name)
        return selected
    
    def get_selected_parameterize_datasets(self):
        """Get list of selected datasets for parameterization."""
        selected = []
        for item in self.param_tree.get_children():
            checkbox_text = self.param_tree.item(item, 'text')
            if checkbox_text == '☑':
                dataset_name = self.param_tree.item(item, 'values')[0]
                selected.append(dataset_name)
        return selected
    
    def log_reconstruct(self, message):
        """Add message to reconstruction log."""
        self.reconstruct_log.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.reconstruct_log.see(tk.END)
        self.root.update_idletasks()
    
    def log_parameterize(self, message):
        """Add message to parameterization log."""
        self.param_log.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.param_log.see(tk.END)
        self.root.update_idletasks()
    
    def run_reconstruction(self):
        """Run 3D reconstruction on selected datasets."""
        if self.is_processing:
            messagebox.showwarning("Warning", "Another operation is already running.")
            return
        
        selected = self.get_selected_reconstruct_datasets()
        if not selected:
            messagebox.showwarning("Warning", "Please select at least one dataset.")
            return
        
        # Confirm
        if not messagebox.askyesno("Confirm", 
                                  f"Reconstruct {len(selected)} dataset(s)?\n\n"
                                  f"Selected: {', '.join(selected)}"):
            return
        
        # Run in background thread
        self.is_processing = True
        self.current_operation = "reconstruction"
        self.reconstruct_btn.config(state='disabled')
        self.reconstruct_progress['maximum'] = len(selected)
        self.reconstruct_progress['value'] = 0
        self.reconstruct_log.delete('1.0', tk.END)
        self.status_var.set(f"Reconstructing {len(selected)} dataset(s)...")
        
        thread = threading.Thread(target=self._run_reconstruction_thread, args=(selected,))
        thread.daemon = True
        thread.start()
    
    def _run_reconstruction_thread(self, datasets):
        """Run reconstruction in background thread."""
        import sys
        import warnings
        
        # Suppress Tkinter cleanup warnings from background threads
        # These are harmless - they occur when Tkinter objects are cleaned up
        # in a background thread, but don't affect functionality
        original_showwarning = warnings.showwarning
        
        def filtered_showwarning(message, category, filename, lineno, file=None, line=None):
            """Filter out Tkinter cleanup warnings."""
            if 'RuntimeError' in str(message) and 'main thread is not in main loop' in str(message):
                return  # Suppress these warnings
            original_showwarning(message, category, filename, lineno, file, line)
        
        warnings.showwarning = filtered_showwarning
        
        try:
            self.log_reconstruct(f"Starting reconstruction of {len(datasets)} dataset(s)...")
            failed = []
            successful = []
            
            for i, dataset_name in enumerate(datasets):
                try:
                    # Update status
                    self.root.after(0, lambda d=dataset_name: self.status_var.set(f"Reconstructing: {d}"))
                    self.log_reconstruct(f"\nProcessing: {dataset_name}")
                    
                    # Get dataset link
                    if dataset_name not in self.dataset_links:
                        self.log_reconstruct(f"  ERROR: Dataset '{dataset_name}' not found in links")
                        failed.append(dataset_name)
                        self._mark_reconstruct_failed(dataset_name)
                        continue
                    
                    dataset_link = self.dataset_links[dataset_name]
                    
                    # Run reconstruction
                    reconstruct_dataset(dataset_name, dataset_link)
                    
                    self.log_reconstruct(f"  [OK] Successfully reconstructed {dataset_name}")
                    successful.append(dataset_name)
                    self._mark_reconstruct_success(dataset_name)
                    
                except Exception as e:
                    self.log_reconstruct(f"  [ERROR] Failed to reconstruct {dataset_name}: {e}")
                    failed.append(dataset_name)
                    self._mark_reconstruct_failed(dataset_name)
                
                # Update progress
                self.root.after(0, lambda v=i+1: setattr(self.reconstruct_progress, 'value', v))
            
            # Final summary
            self.log_reconstruct(f"\n{'='*60}")
            self.log_reconstruct(f"Reconstruction complete!")
            self.log_reconstruct(f"  Successful: {len(successful)}")
            self.log_reconstruct(f"  Failed: {len(failed)}")
            if failed:
                self.log_reconstruct(f"  Failed datasets: {', '.join(failed)}")
            
            # Refresh and re-enable
            self.root.after(0, self._reconstruction_complete)
        finally:
            # Restore original warning handler
            warnings.showwarning = original_showwarning
    
    def _mark_reconstruct_success(self, dataset_name):
        """Mark dataset as successfully reconstructed."""
        for item in self.reconstruct_tree.get_children():
            if self.reconstruct_tree.item(item, 'values')[0] == dataset_name:
                # Uncheck checkbox and update status
                self.reconstruct_tree.item(item, text='☐')
                values = list(self.reconstruct_tree.item(item, 'values'))
                values[4] = '[OK] Reconstructed'  # Status is the 5th column (index 4)
                self.reconstruct_tree.item(item, values=values)
                self.reconstruct_tree.item(item, tags=('green',))
                break
    
    def _mark_reconstruct_failed(self, dataset_name):
        """Mark dataset as failed."""
        for item in self.reconstruct_tree.get_children():
            if self.reconstruct_tree.item(item, 'values')[0] == dataset_name:
                # Update status
                values = list(self.reconstruct_tree.item(item, 'values'))
                values[4] = '[ERROR] Failed'  # Status is the 5th column (index 4)
                self.reconstruct_tree.item(item, values=values)
                self.reconstruct_tree.item(item, tags=('red',))
                break
    
    def _reconstruction_complete(self):
        """Called when reconstruction is complete."""
        self.is_processing = False
        self.current_operation = None
        self.reconstruct_btn.config(state='normal')
        self.status_var.set("Ready")
        self.refresh_reconstruction_list()
    
    def run_parameterization(self):
        """Run parameterization on selected datasets."""
        if self.is_processing:
            messagebox.showwarning("Warning", "Another operation is already running.")
            return
        
        selected = self.get_selected_parameterize_datasets()
        if not selected:
            messagebox.showwarning("Warning", "Please select at least one dataset.")
            return
        
        # Confirm (warn about overwriting)
        msg = f"Parameterize {len(selected)} dataset(s)?\n\n"
        msg += f"Selected: {', '.join(selected)}\n\n"
        overwrite = [d for d in selected if d in self.parameterized]
        if overwrite:
            msg += f"WARNING: {len(overwrite)} dataset(s) will be overwritten:\n"
            msg += f"{', '.join(overwrite)}\n\n"
        msg += "Continue?"
        
        if not messagebox.askyesno("Confirm", msg):
            return
        
        # Get options
        trim_enabled = self.trim_enable_var.get()
        trim_condition = self.trim_condition_var.get() if trim_enabled else 'none'
        
        # Run in background thread
        self.is_processing = True
        self.current_operation = "parameterization"
        self.param_btn.config(state='disabled')
        self.param_progress['maximum'] = len(selected)
        self.param_progress['value'] = 0
        self.param_log.delete('1.0', tk.END)
        self.status_var.set(f"Parameterizing {len(selected)} dataset(s)...")
        
        thread = threading.Thread(target=self._run_parameterization_thread, 
                                 args=(selected, trim_enabled, trim_condition))
        thread.daemon = True
        thread.start()
    
    def _run_parameterization_thread(self, datasets, trim_enabled, trim_condition):
        """Run parameterization in background thread."""
        self.log_parameterize(f"Starting parameterization of {len(datasets)} dataset(s)...")
        self.log_parameterize(f"Trim enabled: {trim_enabled}, Condition: {trim_condition}")
        failed = []
        successful = []
        
        config = load_config()
        dataset_links = load_links_param()
        
        for i, dataset_name in enumerate(datasets):
            try:
                # Update status
                self.root.after(0, lambda d=dataset_name: self.status_var.set(f"Parameterizing: {d}"))
                self.log_parameterize(f"\nProcessing: {dataset_name}")
                
                # Load data
                data = load_3d_data(dataset_name)
                species = detect_species(dataset_name, dataset_links)
                species_data = load_species_data(species)
                
                if data['branch_info'] is None:
                    self.log_parameterize(f"  ERROR: No branch info found")
                    failed.append(dataset_name)
                    self._mark_param_failed(dataset_name)
                    continue
                
                self.log_parameterize(f"  Loaded {data['frames']} frames, Species: {species}")
                
                # Calculate normalization
                normalization_factor, size_measurements = calculate_ant_size_normalization(
                    data, config['normalization_method'],
                    config['thorax_length_points'], config['body_length_points']
                )
                self.log_parameterize(f"  Normalization factor: {normalization_factor:.3f} mm")
                
                # Trim if enabled and calculate duty factors
                start_frame, end_frame, cycle_status = None, None, None
                if trim_enabled and trim_condition == 'gait_cycle':
                    try:
                        self.log_parameterize(f"  Trimming to gait cycle...")
                        data, duty_factor_df, start_frame, end_frame, cycle_status = trim_data(
                            data, config, data['branch_info']
                        )
                        if start_frame is not None:
                            self.log_parameterize(f"  Trimmed to frames {start_frame}-{end_frame} ({cycle_status})")
                            self.log_parameterize(f"  Trimmed length: {data['frames']} frames")
                        else:
                            self.log_parameterize(f"  Warning: Could not find valid gait cycle: {cycle_status}")
                            self.log_parameterize(f"  Using full dataset")
                            # Calculate duty factor for full dataset
                            duty_factor_data = {'Frame': [], 'Time': []}
                            for foot in FOOT_POINTS:
                                duty_factor_data[f'foot_{foot}_attached'] = []
                            for frame in range(data['frames']):
                                duty_factor_data['Frame'].append(frame)
                                duty_factor_data['Time'].append(frame / config['frame_rate'])
                                for foot in FOOT_POINTS:
                                    is_attached = check_foot_attachment(
                                        data, frame, foot, data['branch_info'],
                                        config['foot_branch_distance'],
                                        config['foot_immobility_threshold'],
                                        config['immobility_frames']
                                    )
                                    duty_factor_data[f'foot_{foot}_attached'].append(is_attached)
                            duty_factor_df = pd.DataFrame(duty_factor_data)
                    except Exception as e:
                        self.log_parameterize(f"  Warning: Trimming failed: {e}, using all frames")
                        # Calculate duty factor for full dataset
                        duty_factor_data = {'Frame': [], 'Time': []}
                        for foot in FOOT_POINTS:
                            duty_factor_data[f'foot_{foot}_attached'] = []
                        for frame in range(data['frames']):
                            duty_factor_data['Frame'].append(frame)
                            duty_factor_data['Time'].append(frame / config['frame_rate'])
                            for foot in FOOT_POINTS:
                                is_attached = check_foot_attachment(
                                    data, frame, foot, data['branch_info'],
                                    config['foot_branch_distance'],
                                    config['foot_immobility_threshold'],
                                    config['immobility_frames']
                                )
                                duty_factor_data[f'foot_{foot}_attached'].append(is_attached)
                        duty_factor_df = pd.DataFrame(duty_factor_data)
                else:
                    # Calculate duty factor for full dataset
                    self.log_parameterize(f"  Calculating foot attachments...")
                    duty_factor_data = {'Frame': [], 'Time': []}
                    for foot in FOOT_POINTS:
                        duty_factor_data[f'foot_{foot}_attached'] = []
                    for frame in range(data['frames']):
                        duty_factor_data['Frame'].append(frame)
                        duty_factor_data['Time'].append(frame / config['frame_rate'])
                        for foot in FOOT_POINTS:
                            is_attached = check_foot_attachment(
                                data, frame, foot, data['branch_info'],
                                config['foot_branch_distance'],
                                config['foot_immobility_threshold'],
                                config['immobility_frames']
                            )
                            duty_factor_data[f'foot_{foot}_attached'].append(is_attached)
                    duty_factor_df = pd.DataFrame(duty_factor_data)
                
                # Calculate parameters
                self.log_parameterize(f"  Calculating parameters...")
                results = calculate_all_parameters(
                    data, duty_factor_df, data['branch_info'], 
                    species_data, config, normalization_factor, dataset_name
                )
                
                # Save
                output_file = save_parameterized_data(
                    dataset_name, data, results, duty_factor_df, data['branch_info'],
                    normalization_factor, size_measurements, config,
                    start_frame, end_frame, cycle_status
                )
                
                self.log_parameterize(f"  [OK] Successfully parameterized {dataset_name}")
                self.log_parameterize(f"  Saved to: {output_file}")
                successful.append(dataset_name)
                self._mark_param_success(dataset_name)
                
            except Exception as e:
                import traceback
                error_msg = str(e)
                self.log_parameterize(f"  [ERROR] Failed to parameterize {dataset_name}: {error_msg}")
                self.log_parameterize(f"  Traceback: {traceback.format_exc()}")
                failed.append(dataset_name)
                self._mark_param_failed(dataset_name)
            
            # Update progress
            self.root.after(0, lambda v=i+1: setattr(self.param_progress, 'value', v))
        
        # Final summary
        self.log_parameterize(f"\n{'='*60}")
        self.log_parameterize(f"Parameterization complete!")
        self.log_parameterize(f"  Successful: {len(successful)}")
        self.log_parameterize(f"  Failed: {len(failed)}")
        if failed:
            self.log_parameterize(f"  Failed datasets: {', '.join(failed)}")
        
        # Refresh and re-enable
        self.root.after(0, self._parameterization_complete)
    
    def _mark_param_success(self, dataset_name):
        """Mark dataset as successfully parameterized."""
        for item in self.param_tree.get_children():
            if self.param_tree.item(item, 'values')[0] == dataset_name:
                # Uncheck checkbox and update status
                self.param_tree.item(item, text='☐')
                values = list(self.param_tree.item(item, 'values'))
                values[3] = '[OK] Parameterized'  # Status is the 4th column (index 3)
                self.param_tree.item(item, values=values)
                self.param_tree.item(item, tags=('green',))
                break
    
    def _mark_param_failed(self, dataset_name):
        """Mark dataset as failed."""
        for item in self.param_tree.get_children():
            if self.param_tree.item(item, 'values')[0] == dataset_name:
                # Update status
                values = list(self.param_tree.item(item, 'values'))
                values[3] = '[ERROR] Failed'  # Status is the 4th column (index 3)
                self.param_tree.item(item, values=values)
                self.param_tree.item(item, tags=('red',))
                break
    
    def _parameterization_complete(self):
        """Called when parameterization is complete."""
        self.is_processing = False
        self.current_operation = None
        self.param_btn.config(state='normal')
        self.status_var.set("Ready")
        self.refresh_parameterization_list()
    
    def on_reconstruct_tree_click(self, event):
        """Handle clicks on reconstruction tree."""
        item = self.reconstruct_tree.identify_row(event.y)
        if not item:
            return
        
        # Check if click is in checkbox column (#0) - first 35 pixels
        column = self.reconstruct_tree.identify_column(event.x)
        if column == '#0' or event.x < 35:  # Checkbox column or first 35 pixels
            current = self.reconstruct_tree.item(item, 'text')
            new_value = '☑' if current == '☐' else '☐'
            self.reconstruct_tree.item(item, text=new_value)
            # Update select all state
            all_selected = all(self.reconstruct_tree.item(child, 'text') == '☑' 
                              for child in self.reconstruct_tree.get_children())
            self.reconstruct_select_all_var.set(all_selected)
            return 'break'  # Prevent default selection behavior
    
    def on_param_tree_click(self, event):
        """Handle clicks on parameterization tree."""
        item = self.param_tree.identify_row(event.y)
        if not item:
            return
        
        # Check if click is in checkbox column (#0) - first 35 pixels
        column = self.param_tree.identify_column(event.x)
        if column == '#0' or event.x < 35:  # Checkbox column or first 35 pixels
            current = self.param_tree.item(item, 'text')
            new_value = '☑' if current == '☐' else '☐'
            self.param_tree.item(item, text=new_value)
            # Update select all state
            all_selected = all(self.param_tree.item(child, 'text') == '☑' 
                             for child in self.param_tree.get_children())
            self.param_select_all_var.set(all_selected)
            return 'break'  # Prevent default selection behavior
    
    def view_interactive_figure(self):
        """Open interactive 3D visualization for selected datasets."""
        print("DEBUG: view_interactive_figure called")  # Debug output
        self.log_parameterize("DEBUG: View Interactive Figure button clicked")
        
        selected = self.get_selected_parameterize_datasets()
        self.log_parameterize(f"DEBUG: Selected datasets: {selected}")
        
        if not selected:
            messagebox.showwarning("Warning", "Please select at least one dataset to visualize.")
            return
        
        # Check if datasets are reconstructed
        missing = [d for d in selected if d not in self.available_3d]
        if missing:
            messagebox.showerror("Error", 
                               f"The following datasets are not yet 3D reconstructed:\n"
                               f"{', '.join(missing)}\n\n"
                               f"Please reconstruct them first.")
            return
        
        self.log_parameterize(f"Starting visualization for: {', '.join(selected)}")
        
        # Run visualization in a separate thread to avoid blocking GUI
        try:
            thread = threading.Thread(target=self._run_visualization_thread, args=(selected,))
            thread.daemon = True
            thread.start()
            self.log_parameterize("Visualization thread started")
        except Exception as e:
            error_msg = f"Failed to start visualization thread: {e}"
            self.log_parameterize(f"ERROR: {error_msg}")
            messagebox.showerror("Error", error_msg)
    
    def _run_visualization_thread(self, datasets):
        """Run visualization in background thread."""
        try:
            # Load data for all selected datasets
            all_data = {}
            all_branch_info = {}
            all_com_data = {}
            all_leg_joints_data = {}
            
            for dataset_name in datasets:
                try:
                    # Load 3D data (includes branch_info, com_data, leg_joints_data if available)
                    data = load_3d_data(dataset_name)
                    all_data[dataset_name] = data
                    
                    # Extract branch info from loaded data (preferred) or load separately
                    if data.get('branch_info'):
                        all_branch_info[dataset_name] = data['branch_info']
                    else:
                        # Fallback: Load branch info from dataset links
                        dataset_links = load_links_param()
                        if dataset_name in dataset_links:
                            branch_set_name = dataset_links[dataset_name].get('branch_set')
                            if branch_set_name:
                                branch_set = load_branch_set(branch_set_name)
                                if branch_set:
                                    all_branch_info[dataset_name] = {
                                        'axis_direction': np.array(branch_set.get('axis_direction', [0, 1, 0])),
                                        'axis_point': np.array(branch_set.get('axis_point', [0, 0, 0])),
                                        'radius': branch_set.get('radius', 0)
                                    }
                    
                    # Extract CoM and leg joint data if available
                    if data.get('com_data'):
                        all_com_data[dataset_name] = data['com_data']
                    if data.get('leg_joints_data'):
                        all_leg_joints_data[dataset_name] = data['leg_joints_data']
                except Exception as e:
                    error_msg = f"Error loading {dataset_name}: {e}"
                    self.root.after(0, lambda msg=error_msg: self.log_parameterize(msg))
                    self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", msg))
                    continue
            
            if not all_data:
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not load any datasets for visualization."))
                return
            
            # Create visualization (must run on main thread for matplotlib)
            if len(all_data) == 1:
                # Single dataset visualization
                dataset_name = list(all_data.keys())[0]
                data_copy = all_data[dataset_name]
                branch_copy = all_branch_info.get(dataset_name)
                name_copy = dataset_name
                print(f"DEBUG: Scheduling visualization for {name_copy}")
                self.root.after(0, lambda: self.log_parameterize(f"DEBUG: About to call _visualize_single_ant for {name_copy}"))
                # Use after_idle to ensure it runs after all pending events
                com_copy = all_com_data.get(dataset_name)
                leg_joints_copy = all_leg_joints_data.get(dataset_name)
                self.root.after_idle(lambda: self._visualize_single_ant(data_copy, branch_copy, name_copy, com_copy, leg_joints_copy))
            else:
                # Multiple datasets visualization
                data_copy = all_data.copy()
                branch_copy = all_branch_info.copy()
                com_copy = all_com_data.copy()
                leg_joints_copy = all_leg_joints_data.copy()
                print("DEBUG: Scheduling multiple ants visualization")
                self.root.after(0, lambda: self.log_parameterize("DEBUG: About to call _visualize_multiple_ants"))
                self.root.after_idle(lambda: self._visualize_multiple_ants(data_copy, branch_copy, com_copy, leg_joints_copy))
                
        except Exception as e:
            import traceback
            error_msg = f"Failed to create visualization: {e}\n{traceback.format_exc()}"
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", msg))
    
    def _visualize_single_ant(self, ant_data, branch_info, dataset_name, com_data=None, leg_joints_data=None):
        """Visualize a single ant dataset interactively."""
        print(f"DEBUG: _visualize_single_ant called for {dataset_name}")
        print(f"DEBUG: Current matplotlib backend: {matplotlib.get_backend()}")
        self.log_parameterize(f"DEBUG: _visualize_single_ant executing for {dataset_name}")
        
        # Ensure we have a GUI backend - Agg is non-interactive!
        current_backend = matplotlib.get_backend()
        if current_backend == 'Agg' or 'Agg' in current_backend:
            print("WARNING: Backend is Agg (non-interactive), trying to switch...")
            self.log_parameterize("WARNING: Switching from non-interactive backend to GUI backend")
            # Need to reload pyplot after switching backend
            import importlib
            try:
                matplotlib.use('TkAgg', force=True)
                importlib.reload(plt)
                print("DEBUG: Switched to TkAgg backend and reloaded pyplot")
            except Exception as e1:
                try:
                    matplotlib.use('Qt5Agg', force=True)
                    importlib.reload(plt)
                    print("DEBUG: Switched to Qt5Agg backend and reloaded pyplot")
                except Exception as e2:
                    self.log_parameterize("ERROR: Cannot switch to GUI backend!")
                    messagebox.showerror("Error", f"Cannot create interactive figure: matplotlib is using non-interactive backend '{current_backend}'. Please install tkinter or PyQt5.")
                    return
        
        try:
            # Create a new figure with explicit window number
            fig = plt.figure(figsize=(18, 10), num=f"3D Visualization: {dataset_name}")
            
            # Create main 3D plot (leave space on left for checkboxes)
            ax = fig.add_axes([0.25, 0.1, 0.7, 0.85], projection='3d')
            
            total_frames = ant_data['frames']
            current_frame = [0]
            
            # Calculate axis limits
            x_min = y_min = z_min = float('inf')
            x_max = y_max = z_max = float('-inf')
            
            for frame in range(total_frames):
                for point in range(1, 17):
                    if point in ant_data['points']:
                        x_min = min(x_min, ant_data['points'][point]['X'][frame])
                        x_max = max(x_max, ant_data['points'][point]['X'][frame])
                        y_min = min(y_min, ant_data['points'][point]['Y'][frame])
                        y_max = max(y_max, ant_data['points'][point]['Y'][frame])
                        z_min = min(z_min, ant_data['points'][point]['Z'][frame])
                        z_max = max(z_max, ant_data['points'][point]['Z'][frame])
            
            # Add padding
            x_pad = 0.2 * (x_max - x_min) if x_max > x_min else 1
            y_pad = 0.2 * (y_max - y_min) if y_max > y_min else 1
            z_pad = 0.2 * (z_max - z_min) if z_max > z_min else 1
            
            x_min -= x_pad
            x_max += x_pad
            y_min -= y_pad
            y_max += y_pad
            z_min -= z_pad
            z_max += z_pad
            
            # Create visibility state dictionary
            visibility = {
                'branch_axis': True,
                'branch_surface': True,
                'body_points': True,
                'body_line': True,
                'tracking_points': True,
                'com_overall': True,
                'com_head': True,
                'com_thorax': True,
                'com_gaster': True,
                'leg_joints': True,
                'leg_connections': True
            }
            
            # Add individual tracking point visibility
            for point in range(1, 17):
                visibility[f'point_{point}'] = True
            
            def update_plot(frame_idx):
                ax.clear()
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_zlim(z_min, z_max)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'{dataset_name} - Frame {frame_idx + 1}/{total_frames}')
                
                # Plot branch if available
                if branch_info:
                    if visibility['branch_axis']:
                        t = np.linspace(-2, 2, 100)
                        line_points = branch_info['axis_point'] + np.outer(t, branch_info['axis_direction'])
                        ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                               '--', color='green', linewidth=1, alpha=0.5)
                    
                    # Plot branch surface (transparent cylinder)
                    if visibility['branch_surface'] and branch_info.get('radius') is not None:
                        radius = branch_info['radius']
                        axis_point = np.array(branch_info['axis_point'])
                        axis_direction = np.array(branch_info['axis_direction'])
                        axis_direction = axis_direction / np.linalg.norm(axis_direction)  # Normalize
                        
                        # Create cylinder along the branch axis
                        # Extend the axis to cover the plot area
                        t_cylinder = np.linspace(-3, 3, 50)
                        axis_length = np.linalg.norm([x_max - x_min, y_max - y_min, z_max - z_min])
                        t_cylinder = np.linspace(-axis_length/2, axis_length/2, 50)
                        
                        # Create cylinder points
                        theta = np.linspace(0, 2*np.pi, 30)
                        z_cyl = np.linspace(0, 1, len(t_cylinder))
                        
                        # Create a circle perpendicular to the axis direction
                        # Find two perpendicular vectors to the axis
                        if abs(axis_direction[2]) < 0.9:
                            perp1 = np.cross(axis_direction, np.array([0, 0, 1]))
                        else:
                            perp1 = np.cross(axis_direction, np.array([1, 0, 0]))
                        perp1 = perp1 / np.linalg.norm(perp1)
                        perp2 = np.cross(axis_direction, perp1)
                        perp2 = perp2 / np.linalg.norm(perp2)
                        
                        # Generate cylinder surface
                        X_cyl = np.zeros((len(t_cylinder), len(theta)))
                        Y_cyl = np.zeros((len(t_cylinder), len(theta)))
                        Z_cyl = np.zeros((len(t_cylinder), len(theta)))
                        
                        for i, t_val in enumerate(t_cylinder):
                            center_point = axis_point + t_val * axis_direction
                            for j, theta_val in enumerate(theta):
                                circle_point = center_point + radius * (np.cos(theta_val) * perp1 + np.sin(theta_val) * perp2)
                                X_cyl[i, j] = circle_point[0]
                                Y_cyl[i, j] = circle_point[1]
                                Z_cyl[i, j] = circle_point[2]
                        
                        # Plot cylinder surface
                        ax.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.2, color='green', shade=False)
                
                # Plot body points (1-4)
                if visibility['body_points'] or visibility['body_line']:
                    body_x = [ant_data['points'][i]['X'][frame_idx] for i in range(1, 5) if i in ant_data['points']]
                    body_y = [ant_data['points'][i]['Y'][frame_idx] for i in range(1, 5) if i in ant_data['points']]
                    body_z = [ant_data['points'][i]['Z'][frame_idx] for i in range(1, 5) if i in ant_data['points']]
                    if body_x:
                        if visibility['body_line']:
                            ax.plot(body_x, body_y, body_z, '-', color='red', linewidth=2, label='Body')
                        if visibility['body_points']:
                            ax.scatter(body_x, body_y, body_z, color='red', s=50)
                
                # Plot all tracking points
                if visibility['tracking_points']:
                    colors = plt.cm.tab20(np.linspace(0, 1, 16))
                    for point in range(1, 17):
                        if visibility[f'point_{point}'] and point in ant_data['points']:
                            x = ant_data['points'][point]['X'][frame_idx]
                            y = ant_data['points'][point]['Y'][frame_idx]
                            z = ant_data['points'][point]['Z'][frame_idx]
                            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                                ax.scatter(x, y, z, color=colors[point-1], s=30, label=f'Point {point}')
                
                # Plot CoM points with red circles
                if com_data is not None:
                    com_labels = ['overall', 'head', 'thorax', 'gaster']
                    com_colors = ['red', 'darkred', 'crimson', 'firebrick']
                    for i, (label, color) in enumerate(zip(com_labels, com_colors)):
                        if visibility[f'com_{label}'] and frame_idx < len(com_data[label]):
                            com_pos = com_data[label][frame_idx]
                            if not np.any(np.isnan(com_pos)):
                                ax.scatter(com_pos[0], com_pos[1], com_pos[2], 
                                         color=color, s=100, marker='o', 
                                         edgecolors='red', linewidths=2, 
                                         label=f'CoM {label.capitalize()}', alpha=0.8)
                
                # Plot leg joints and connect legs
                if leg_joints_data is not None:
                    # Leg mapping: (foot_point, femur_tibia_point, joint_name)
                    leg_mappings = [
                        (8, 5, 'front_left'),   # Left front
                        (9, 6, 'mid_left'),     # Left middle
                        (10, 7, 'hind_left'),   # Left hind
                        (14, 11, 'front_right'), # Right front
                        (15, 12, 'mid_right'),  # Right middle
                        (16, 13, 'hind_right')  # Right hind
                    ]
                    
                    for foot_point, femur_tibia_point, joint_name in leg_mappings:
                        # Get foot position
                        if foot_point in ant_data['points']:
                            foot_x = ant_data['points'][foot_point]['X'][frame_idx]
                            foot_y = ant_data['points'][foot_point]['Y'][frame_idx]
                            foot_z = ant_data['points'][foot_point]['Z'][frame_idx]
                            
                            # Get femur-tibia joint position
                            if femur_tibia_point in ant_data['points']:
                                ft_x = ant_data['points'][femur_tibia_point]['X'][frame_idx]
                                ft_y = ant_data['points'][femur_tibia_point]['Y'][frame_idx]
                                ft_z = ant_data['points'][femur_tibia_point]['Z'][frame_idx]
                                
                                # Get leg body joint position
                                if frame_idx < len(leg_joints_data[joint_name]):
                                    joint_pos = leg_joints_data[joint_name][frame_idx]
                                    
                                    # Check if all points are valid
                                    if (not (np.isnan(foot_x) or np.isnan(foot_y) or np.isnan(foot_z)) and
                                        not (np.isnan(ft_x) or np.isnan(ft_y) or np.isnan(ft_z)) and
                                        not np.any(np.isnan(joint_pos))):
                                        
                                        # Plot leg body joint
                                        if visibility['leg_joints']:
                                            ax.scatter(joint_pos[0], joint_pos[1], joint_pos[2], 
                                                     color='blue', s=60, marker='s')
                                        
                                        # Draw connections: foot -> femur-tibia -> leg body joint
                                        if visibility['leg_connections']:
                                            # Foot to femur-tibia
                                            ax.plot([foot_x, ft_x], [foot_y, ft_y], [foot_z, ft_z],
                                                   color='purple', linewidth=2, alpha=0.7)
                                            # Femur-tibia to leg body joint
                                            ax.plot([ft_x, joint_pos[0]], [ft_y, joint_pos[1]], [ft_z, joint_pos[2]],
                                                   color='purple', linewidth=2, alpha=0.7)
                
                # No legend - using interactive checkboxes instead
                plt.draw()
            
            def on_key(event):
                if event.key == 'right' or event.key == 'd':
                    current_frame[0] = min(current_frame[0] + 1, total_frames - 1)
                    update_plot(current_frame[0])
                elif event.key == 'left' or event.key == 'a':
                    current_frame[0] = max(current_frame[0] - 1, 0)
                    update_plot(current_frame[0])
            
            # Create custom interactive legend/checkbox panel on the left
            legend_ax = fig.add_axes([0.02, 0.05, 0.22, 0.9])
            legend_ax.set_axis_off()
            legend_ax.set_xlim(0, 1)
            legend_ax.set_ylim(0, 1)
            
            # Store legend items for click detection
            legend_items = []
            y_pos = 0.98
            line_height = 0.025
            
            # Get colors for tracking points
            colors = plt.cm.tab20(np.linspace(0, 1, 16))
            
            def create_legend_item(y, symbol_type, color, marker, label, visibility_key):
                """Create a clickable legend item."""
                # Create symbol
                if symbol_type == 'line':
                    # Draw line symbol
                    legend_ax.plot([0.02, 0.08], [y, y], color=color, linewidth=2, linestyle='--' if 'Branch' in label else '-')
                elif symbol_type == 'scatter':
                    # Draw scatter symbol
                    legend_ax.scatter([0.05], [y], color=color, s=50 if 'CoM' in label else 30, marker=marker, 
                                     edgecolors='red' if 'CoM' in label else None, linewidths=2 if 'CoM' in label else 0)
                
                # Create text label
                alpha = 0.3 if not visibility[visibility_key] else 1.0
                text = legend_ax.text(0.12, y, label, fontsize=9, verticalalignment='center',
                                     alpha=alpha, color='gray' if not visibility[visibility_key] else 'black')
                
                # Create invisible clickable rectangle
                rect = plt.Rectangle((0, y - line_height/2), 0.22, line_height, 
                                    transform=legend_ax.transData, alpha=0, picker=True)
                legend_ax.add_patch(rect)
                
                return {'rect': rect, 'text': text, 'symbol_type': symbol_type, 'color': color, 
                       'marker': marker, 'y': y, 'visibility_key': visibility_key, 'label': label}
            
            # Add legend items in order matching the plot
            # Branch Axis
            if branch_info:
                item = create_legend_item(y_pos, 'line', 'green', None, 'Branch Axis', 'branch_axis')
                legend_items.append(item)
                y_pos -= line_height * 1.5
                
                # Branch Surface
                if branch_info.get('radius') is not None:
                    # Create a filled rectangle to represent the surface
                    legend_ax.add_patch(plt.Rectangle((0.02, y_pos - line_height/4), 0.06, line_height/2, 
                                                     facecolor='green', alpha=0.2, edgecolor='green'))
                    text = legend_ax.text(0.12, y_pos, 'Branch Surface', fontsize=9, verticalalignment='center',
                                         alpha=1.0, color='black')
                    rect = plt.Rectangle((0, y_pos - line_height/2), 0.22, line_height, 
                                        transform=legend_ax.transData, alpha=0, picker=True)
                    legend_ax.add_patch(rect)
                    legend_items.append({'rect': rect, 'text': text, 'symbol_type': 'surface', 
                                        'color': 'green', 'marker': None, 'y': y_pos, 
                                        'visibility_key': 'branch_surface', 'label': 'Branch Surface'})
                    y_pos -= line_height * 1.5
            
            # Body Line
            item = create_legend_item(y_pos, 'line', 'red', None, 'Body', 'body_line')
            legend_items.append(item)
            y_pos -= line_height * 1.5
            
            # Body Points (shown as red scatter)
            item = create_legend_item(y_pos, 'scatter', 'red', 'o', 'Body Points', 'body_points')
            legend_items.append(item)
            y_pos -= line_height * 1.5
            
            # CoM points
            if com_data is not None:
                com_labels = ['overall', 'head', 'thorax', 'gaster']
                com_colors = ['red', 'darkred', 'crimson', 'firebrick']
                for label, color in zip(com_labels, com_colors):
                    item = create_legend_item(y_pos, 'scatter', color, 'o', f'CoM {label.capitalize()}', f'com_{label}')
                    legend_items.append(item)
                    y_pos -= line_height * 1.5
            
            # Body-leg joints
            if leg_joints_data is not None:
                item = create_legend_item(y_pos, 'scatter', 'blue', 's', 'Body-leg joints', 'leg_joints')
                legend_items.append(item)
                y_pos -= line_height * 1.5
            
            # Leg Connections
            if leg_joints_data is not None:
                item = create_legend_item(y_pos, 'line', 'purple', None, 'Leg Connections', 'leg_connections')
                legend_items.append(item)
                y_pos -= line_height * 1.5
            
            # Tracking Points
            item = create_legend_item(y_pos, 'scatter', 'gray', 'o', 'Tracking Points', 'tracking_points')
            legend_items.append(item)
            y_pos -= line_height * 1.5
            
            # Individual tracking points
            for point in range(1, 17):
                if point in ant_data['points']:
                    item = create_legend_item(y_pos, 'scatter', colors[point-1], 'o', f'Point {point}', f'point_{point}')
                    legend_items.append(item)
                    y_pos -= line_height * 1.5
            
            def redraw_legend():
                """Redraw the legend with current visibility states."""
                legend_ax.clear()
                legend_ax.set_axis_off()
                legend_ax.set_xlim(0, 1)
                legend_ax.set_ylim(0, 1)
                
                y_pos_recreate = 0.98
                for item in legend_items:
                    vis_key = item['visibility_key']
                    is_visible = visibility[vis_key]
                    alpha_recreate = 0.3 if not is_visible else 1.0
                    
                    if item['symbol_type'] == 'line':
                        legend_ax.plot([0.02, 0.08], [y_pos_recreate, y_pos_recreate], 
                                     color=item['color'], linewidth=2, 
                                     linestyle='--' if 'Branch' in item['label'] else '-',
                                     alpha=alpha_recreate)
                    elif item['symbol_type'] == 'scatter':
                        legend_ax.scatter([0.05], [y_pos_recreate], color=item['color'], 
                                        s=50 if 'CoM' in item['label'] or 'Body Points' in item['label'] or 'Body-leg' in item['label'] else 30, 
                                        marker=item['marker'],
                                        edgecolors='red' if 'CoM' in item['label'] else None, 
                                        linewidths=2 if 'CoM' in item['label'] else 0,
                                        alpha=alpha_recreate)
                    elif item['symbol_type'] == 'surface':
                        # Draw filled rectangle for surface
                        legend_ax.add_patch(plt.Rectangle((0.02, y_pos_recreate - line_height/4), 0.06, line_height/2, 
                                                         facecolor=item['color'], alpha=0.2 * alpha_recreate, 
                                                         edgecolor=item['color'], linewidth=1))
                    
                    legend_ax.text(0.12, y_pos_recreate, item['label'], fontsize=9, 
                                 verticalalignment='center', alpha=alpha_recreate,
                                 color='gray' if not is_visible else 'black')
                    
                    # Recreate clickable rectangle
                    rect_new = plt.Rectangle((0, y_pos_recreate - line_height/2), 0.22, line_height, 
                                            transform=legend_ax.transData, alpha=0, picker=True)
                    legend_ax.add_patch(rect_new)
                    item['rect'] = rect_new
                    item['y'] = y_pos_recreate
                    
                    y_pos_recreate -= line_height * 1.5
            
            def on_legend_click(event):
                """Handle clicks on legend items."""
                if event.inaxes != legend_ax:
                    return
                
                # Get click coordinates in legend axes
                x_click, y_click = event.xdata, event.ydata
                if x_click is None or y_click is None:
                    return
                
                # Find which item was clicked
                for item in legend_items:
                    y_item = item['y']
                    # Check if click is within the item's vertical range
                    if abs(y_click - y_item) < line_height:
                        # Toggle visibility
                        visibility_key = item['visibility_key']
                        visibility[visibility_key] = not visibility[visibility_key]
                        
                        # Redraw legend and plot
                        redraw_legend()
                        update_plot(current_frame[0])
                        fig.canvas.draw()
                        break
            
            fig.canvas.mpl_connect('button_press_event', on_legend_click)
            
            fig.canvas.mpl_connect('key_press_event', on_key)
            update_plot(0)
            
            plt.figtext(0.27, 0.02, 'Use Left/Right arrow keys or A/D to navigate frames', 
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            print(f"DEBUG: About to show figure for {dataset_name}")
            self.log_parameterize(f"DEBUG: Creating matplotlib figure for {dataset_name}")
            
            # Ensure interactive mode is on
            plt.ion()  # Turn on interactive mode
            
            # Force a draw
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Show the figure - try different methods
            try:
                # Method 1: Standard show
                plt.show(block=False)
                
                # Method 2: Try to get the window and show it explicitly
                if hasattr(fig.canvas, 'manager') and fig.canvas.manager is not None:
                    manager = fig.canvas.manager
                    print(f"DEBUG: Window manager type: {type(manager)}")
                    
                    # Try to get the window and show it
                    if hasattr(manager, 'window'):
                        window = manager.window
                        # For TkAgg (Tkinter window)
                        if hasattr(window, 'deiconify'):
                            window.deiconify()  # Show if hidden
                            window.lift()  # Bring to front
                            window.focus_force()  # Force focus
                            print(f"DEBUG: Tk window shown: {window.winfo_viewable() if hasattr(window, 'winfo_viewable') else 'N/A'}")
                        # For QtAgg (Qt window)
                        elif hasattr(window, 'show'):
                            window.show()
                            window.raise_()
                            window.activateWindow()
                            print(f"DEBUG: Qt window shown")
                    
                    # For TkAgg backend - try show method
                    if hasattr(manager, 'show'):
                        manager.show()
                    
                    # For QtAgg backend - try different approach
                    if 'Qt' in str(type(manager)) or 'qt' in str(type(manager)).lower():
                        try:
                            if hasattr(manager, 'window'):
                                qt_window = manager.window
                                qt_window.show()
                                qt_window.raise_()
                                if hasattr(qt_window, 'activateWindow'):
                                    qt_window.activateWindow()
                                print(f"DEBUG: Qt window activated")
                        except Exception as qt_e:
                            print(f"DEBUG: Qt window error: {qt_e}")
                
                # Force update
                fig.canvas.draw_idle()
                self.root.update()  # Process Tkinter events
                
            except Exception as e:
                print(f"DEBUG: Error showing figure: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"DEBUG: Figure shown for {dataset_name}, backend: {matplotlib.get_backend()}")
            self.log_parameterize(f"DEBUG: Figure window should be visible for {dataset_name} (backend: {matplotlib.get_backend()})")
            
            # Final attempt: try plt.show() again with a small delay and ensure window is visible
            def show_window():
                try:
                    plt.show(block=False)
                    if hasattr(fig.canvas, 'manager') and hasattr(fig.canvas.manager, 'window'):
                        fig.canvas.manager.window.deiconify()
                        fig.canvas.manager.window.lift()
                        fig.canvas.manager.window.focus_force()
                    fig.canvas.draw()
                except Exception as e:
                    print(f"DEBUG: Error in delayed show: {e}")
            
            self.root.after(100, show_window)
            self.root.after(500, show_window)  # Try again after 500ms
            
        except Exception as e:
            import traceback
            error_msg = f"Error in visualization: {e}\n{traceback.format_exc()}"
            print(f"DEBUG ERROR: {error_msg}")
            messagebox.showerror("Visualization Error", error_msg)
    
    def _visualize_multiple_ants(self, all_data, all_branch_info, all_com_data=None, all_leg_joints_data=None):
        """Visualize multiple ant datasets with ability to switch between them."""
        # Ensure we have a GUI backend - Agg is non-interactive!
        current_backend = matplotlib.get_backend()
        if current_backend == 'Agg' or 'Agg' in current_backend:
            print("WARNING: Backend is Agg (non-interactive), trying to switch...")
            self.log_parameterize("WARNING: Switching from non-interactive backend to GUI backend")
            # Need to reload pyplot after switching backend
            import importlib
            try:
                matplotlib.use('TkAgg', force=True)
                importlib.reload(plt)
                print("DEBUG: Switched to TkAgg backend and reloaded pyplot")
            except Exception as e1:
                try:
                    matplotlib.use('Qt5Agg', force=True)
                    importlib.reload(plt)
                    print("DEBUG: Switched to Qt5Agg backend and reloaded pyplot")
                except Exception as e2:
                    self.log_parameterize("ERROR: Cannot switch to GUI backend!")
                    messagebox.showerror("Error", f"Cannot create interactive figure: matplotlib is using non-interactive backend '{current_backend}'. Please install tkinter or PyQt5.")
                    return
        
        # Create figure with explicit window management (leave space for legend)
        dataset_names = list(all_data.keys())
        fig = plt.figure(figsize=(18, 10), num=f"3D Visualization: Multiple Ants")
        
        # Create main 3D plot (leave space on left for checkboxes)
        ax = fig.add_axes([0.25, 0.1, 0.7, 0.85], projection='3d')
        
        current_ant_idx = [0]
        current_frame = [0]
        
        # Calculate global axis limits
        x_min = y_min = z_min = float('inf')
        x_max = y_max = z_max = float('-inf')
        
        for dataset_name, data in all_data.items():
            for frame in range(data['frames']):
                for point in range(1, 17):
                    if point in data['points']:
                        x_min = min(x_min, data['points'][point]['X'][frame])
                        x_max = max(x_max, data['points'][point]['X'][frame])
                        y_min = min(y_min, data['points'][point]['Y'][frame])
                        y_max = max(y_max, data['points'][point]['Y'][frame])
                        z_min = min(z_min, data['points'][point]['Z'][frame])
                        z_max = max(z_max, data['points'][point]['Z'][frame])
        
        # Add padding
        x_pad = 0.2 * (x_max - x_min) if x_max > x_min else 1
        y_pad = 0.2 * (y_max - y_min) if y_max > y_min else 1
        z_pad = 0.2 * (z_max - z_min) if z_max > z_min else 1
        
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad
        z_min -= z_pad
        z_max += z_pad
        
        # Create visibility state dictionary
        visibility = {
            'branch_axis': True,
            'branch_surface': True,
            'body_points': True,
            'body_line': True,
            'tracking_points': True,
            'com_overall': True,
            'com_head': True,
            'com_thorax': True,
            'com_gaster': True,
            'leg_joints': True,
            'leg_connections': True
        }
        
        # Add individual tracking point visibility
        for point in range(1, 17):
            visibility[f'point_{point}'] = True
        
        def update_plot():
            ax.clear()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            dataset_name = dataset_names[current_ant_idx[0]]
            ant_data = all_data[dataset_name]
            frame_idx = current_frame[0]
            total_frames = ant_data['frames']
            
            ax.set_title(f'{dataset_name} ({current_ant_idx[0] + 1}/{len(dataset_names)}) - Frame {frame_idx + 1}/{total_frames}')
            
            # Get current dataset's branch, CoM, and leg joints data
            branch_info = all_branch_info.get(dataset_name)
            com_data = all_com_data.get(dataset_name) if all_com_data else None
            leg_joints_data = all_leg_joints_data.get(dataset_name) if all_leg_joints_data else None
            
            # Plot branch if available
            if branch_info:
                if visibility['branch_axis']:
                    t = np.linspace(-2, 2, 100)
                    line_points = branch_info['axis_point'] + np.outer(t, branch_info['axis_direction'])
                    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                           '--', color='green', linewidth=1, alpha=0.5)
                
                # Plot branch surface (transparent cylinder)
                if visibility['branch_surface'] and branch_info.get('radius') is not None:
                    radius = branch_info['radius']
                    axis_point = np.array(branch_info['axis_point'])
                    axis_direction = np.array(branch_info['axis_direction'])
                    axis_direction = axis_direction / np.linalg.norm(axis_direction)  # Normalize
                    
                    # Create cylinder along the branch axis
                    axis_length = np.linalg.norm([x_max - x_min, y_max - y_min, z_max - z_min])
                    t_cylinder = np.linspace(-axis_length/2, axis_length/2, 50)
                    
                    # Create cylinder points
                    theta = np.linspace(0, 2*np.pi, 30)
                    
                    # Find two perpendicular vectors to the axis
                    if abs(axis_direction[2]) < 0.9:
                        perp1 = np.cross(axis_direction, np.array([0, 0, 1]))
                    else:
                        perp1 = np.cross(axis_direction, np.array([1, 0, 0]))
                    perp1 = perp1 / np.linalg.norm(perp1)
                    perp2 = np.cross(axis_direction, perp1)
                    perp2 = perp2 / np.linalg.norm(perp2)
                    
                    # Generate cylinder surface
                    X_cyl = np.zeros((len(t_cylinder), len(theta)))
                    Y_cyl = np.zeros((len(t_cylinder), len(theta)))
                    Z_cyl = np.zeros((len(t_cylinder), len(theta)))
                    
                    for i, t_val in enumerate(t_cylinder):
                        center_point = axis_point + t_val * axis_direction
                        for j, theta_val in enumerate(theta):
                            circle_point = center_point + radius * (np.cos(theta_val) * perp1 + np.sin(theta_val) * perp2)
                            X_cyl[i, j] = circle_point[0]
                            Y_cyl[i, j] = circle_point[1]
                            Z_cyl[i, j] = circle_point[2]
                    
                    # Plot cylinder surface
                    ax.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.2, color='green', shade=False)
            
            # Plot body points (1-4)
            if visibility['body_points'] or visibility['body_line']:
                body_x = [ant_data['points'][i]['X'][frame_idx] for i in range(1, 5) if i in ant_data['points']]
                body_y = [ant_data['points'][i]['Y'][frame_idx] for i in range(1, 5) if i in ant_data['points']]
                body_z = [ant_data['points'][i]['Z'][frame_idx] for i in range(1, 5) if i in ant_data['points']]
                if body_x:
                    if visibility['body_line']:
                        ax.plot(body_x, body_y, body_z, '-', color='red', linewidth=2)
                    if visibility['body_points']:
                        ax.scatter(body_x, body_y, body_z, color='red', s=50)
            
            # Plot all tracking points
            if visibility['tracking_points']:
                colors = plt.cm.tab20(np.linspace(0, 1, 16))
                for point in range(1, 17):
                    if visibility[f'point_{point}'] and point in ant_data['points']:
                        x = ant_data['points'][point]['X'][frame_idx]
                        y = ant_data['points'][point]['Y'][frame_idx]
                        z = ant_data['points'][point]['Z'][frame_idx]
                        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                            ax.scatter(x, y, z, color=colors[point-1], s=30)
            
            # Plot CoM points with red circles
            if com_data is not None:
                com_labels = ['overall', 'head', 'thorax', 'gaster']
                com_colors = ['red', 'darkred', 'crimson', 'firebrick']
                for i, (label, color) in enumerate(zip(com_labels, com_colors)):
                    if visibility[f'com_{label}'] and frame_idx < len(com_data[label]):
                        com_pos = com_data[label][frame_idx]
                        if not np.any(np.isnan(com_pos)):
                            ax.scatter(com_pos[0], com_pos[1], com_pos[2], 
                                     color=color, s=100, marker='o', 
                                     edgecolors='red', linewidths=2, alpha=0.8)
            
            # Plot leg joints and connect legs
            if leg_joints_data is not None:
                # Leg mapping: (foot_point, femur_tibia_point, joint_name)
                leg_mappings = [
                    (8, 5, 'front_left'),   # Left front
                    (9, 6, 'mid_left'),     # Left middle
                    (10, 7, 'hind_left'),   # Left hind
                    (14, 11, 'front_right'), # Right front
                    (15, 12, 'mid_right'),  # Right middle
                    (16, 13, 'hind_right')  # Right hind
                ]
                
                for foot_point, femur_tibia_point, joint_name in leg_mappings:
                    # Get foot position
                    if foot_point in ant_data['points']:
                        foot_x = ant_data['points'][foot_point]['X'][frame_idx]
                        foot_y = ant_data['points'][foot_point]['Y'][frame_idx]
                        foot_z = ant_data['points'][foot_point]['Z'][frame_idx]
                        
                        # Get femur-tibia joint position
                        if femur_tibia_point in ant_data['points']:
                            ft_x = ant_data['points'][femur_tibia_point]['X'][frame_idx]
                            ft_y = ant_data['points'][femur_tibia_point]['Y'][frame_idx]
                            ft_z = ant_data['points'][femur_tibia_point]['Z'][frame_idx]
                            
                            # Get leg body joint position
                            if frame_idx < len(leg_joints_data[joint_name]):
                                joint_pos = leg_joints_data[joint_name][frame_idx]
                                
                                # Check if all points are valid
                                if (not (np.isnan(foot_x) or np.isnan(foot_y) or np.isnan(foot_z)) and
                                    not (np.isnan(ft_x) or np.isnan(ft_y) or np.isnan(ft_z)) and
                                    not np.any(np.isnan(joint_pos))):
                                    
                                    # Plot leg body joint
                                    if visibility['leg_joints']:
                                        ax.scatter(joint_pos[0], joint_pos[1], joint_pos[2], 
                                                 color='blue', s=60, marker='s')
                                    
                                    # Draw connections: foot -> femur-tibia -> leg body joint
                                    if visibility['leg_connections']:
                                        # Foot to femur-tibia
                                        ax.plot([foot_x, ft_x], [foot_y, ft_y], [foot_z, ft_z],
                                               color='purple', linewidth=2, alpha=0.7)
                                        # Femur-tibia to leg body joint
                                        ax.plot([ft_x, joint_pos[0]], [ft_y, joint_pos[1]], [ft_z, joint_pos[2]],
                                               color='purple', linewidth=2, alpha=0.7)
            
            plt.draw()
        
        def on_key(event):
            dataset_name = dataset_names[current_ant_idx[0]]
            ant_data = all_data[dataset_name]
            total_frames = ant_data['frames']
            
            if event.key == 'right' or event.key == 'd':
                current_frame[0] = min(current_frame[0] + 1, total_frames - 1)
                update_plot()
            elif event.key == 'left' or event.key == 'a':
                current_frame[0] = max(current_frame[0] - 1, 0)
                update_plot()
            elif event.key == 'up' or event.key == 'w':
                current_ant_idx[0] = (current_ant_idx[0] + 1) % len(dataset_names)
                current_frame[0] = 0
                update_plot()
            elif event.key == 'down' or event.key == 's':
                current_ant_idx[0] = (current_ant_idx[0] - 1) % len(dataset_names)
                current_frame[0] = 0
                update_plot()
        
        # Create custom interactive legend/checkbox panel on the left
        legend_ax = fig.add_axes([0.02, 0.05, 0.22, 0.9])
        legend_ax.set_axis_off()
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, 1)
        
        # Store legend items for click detection
        legend_items = []
        y_pos = 0.98
        line_height = 0.025
        
        # Get colors for tracking points
        colors = plt.cm.tab20(np.linspace(0, 1, 16))
        
        def create_legend_item(y, symbol_type, color, marker, label, visibility_key):
            """Create a clickable legend item."""
            # Create symbol
            if symbol_type == 'line':
                # Draw line symbol
                legend_ax.plot([0.02, 0.08], [y, y], color=color, linewidth=2, linestyle='--' if 'Branch' in label else '-')
            elif symbol_type == 'scatter':
                # Draw scatter symbol
                legend_ax.scatter([0.05], [y], color=color, s=50 if 'CoM' in label else 30, marker=marker, 
                                 edgecolors='red' if 'CoM' in label else None, linewidths=2 if 'CoM' in label else 0)
            
            # Create text label
            alpha = 0.3 if not visibility[visibility_key] else 1.0
            text = legend_ax.text(0.12, y, label, fontsize=9, verticalalignment='center',
                                 alpha=alpha, color='gray' if not visibility[visibility_key] else 'black')
            
            # Create invisible clickable rectangle
            rect = plt.Rectangle((0, y - line_height/2), 0.22, line_height, 
                                transform=legend_ax.transData, alpha=0, picker=True)
            legend_ax.add_patch(rect)
            
            return {'rect': rect, 'text': text, 'symbol_type': symbol_type, 'color': color, 
                   'marker': marker, 'y': y, 'visibility_key': visibility_key, 'label': label}
        
        # Get first dataset to determine what to show in legend
        first_dataset = dataset_names[0]
        first_branch_info = all_branch_info.get(first_dataset)
        first_com_data = all_com_data.get(first_dataset) if all_com_data else None
        first_leg_joints_data = all_leg_joints_data.get(first_dataset) if all_leg_joints_data else None
        
        # Add legend items in order matching the plot
        # Branch Axis
        if first_branch_info:
            item = create_legend_item(y_pos, 'line', 'green', None, 'Branch Axis', 'branch_axis')
            legend_items.append(item)
            y_pos -= line_height * 1.5
            
            # Branch Surface
            if first_branch_info.get('radius') is not None:
                # Create a filled rectangle to represent the surface
                legend_ax.add_patch(plt.Rectangle((0.02, y_pos - line_height/4), 0.06, line_height/2, 
                                                 facecolor='green', alpha=0.2, edgecolor='green'))
                text = legend_ax.text(0.12, y_pos, 'Branch Surface', fontsize=9, verticalalignment='center',
                                     alpha=1.0, color='black')
                rect = plt.Rectangle((0, y_pos - line_height/2), 0.22, line_height, 
                                    transform=legend_ax.transData, alpha=0, picker=True)
                legend_ax.add_patch(rect)
                legend_items.append({'rect': rect, 'text': text, 'symbol_type': 'surface', 
                                    'color': 'green', 'marker': None, 'y': y_pos, 
                                    'visibility_key': 'branch_surface', 'label': 'Branch Surface'})
                y_pos -= line_height * 1.5
        
        # Body Line
        item = create_legend_item(y_pos, 'line', 'red', None, 'Body', 'body_line')
        legend_items.append(item)
        y_pos -= line_height * 1.5
        
        # Body Points (shown as red scatter)
        item = create_legend_item(y_pos, 'scatter', 'red', 'o', 'Body Points', 'body_points')
        legend_items.append(item)
        y_pos -= line_height * 1.5
        
        # CoM points
        if first_com_data is not None:
            com_labels = ['overall', 'head', 'thorax', 'gaster']
            com_colors = ['red', 'darkred', 'crimson', 'firebrick']
            for label, color in zip(com_labels, com_colors):
                item = create_legend_item(y_pos, 'scatter', color, 'o', f'CoM {label.capitalize()}', f'com_{label}')
                legend_items.append(item)
                y_pos -= line_height * 1.5
        
        # Body-leg joints
        if first_leg_joints_data is not None:
            item = create_legend_item(y_pos, 'scatter', 'blue', 's', 'Body-leg joints', 'leg_joints')
            legend_items.append(item)
            y_pos -= line_height * 1.5
        
        # Leg Connections
        if first_leg_joints_data is not None:
            item = create_legend_item(y_pos, 'line', 'purple', None, 'Leg Connections', 'leg_connections')
            legend_items.append(item)
            y_pos -= line_height * 1.5
        
        # Tracking Points
        item = create_legend_item(y_pos, 'scatter', 'gray', 'o', 'Tracking Points', 'tracking_points')
        legend_items.append(item)
        y_pos -= line_height * 1.5
        
        # Individual tracking points
        if first_dataset in all_data:
            first_ant_data = all_data[first_dataset]
            for point in range(1, 17):
                if point in first_ant_data['points']:
                    item = create_legend_item(y_pos, 'scatter', colors[point-1], 'o', f'Point {point}', f'point_{point}')
                    legend_items.append(item)
                    y_pos -= line_height * 1.5
        
        def redraw_legend():
            """Redraw the legend with current visibility states."""
            legend_ax.clear()
            legend_ax.set_axis_off()
            legend_ax.set_xlim(0, 1)
            legend_ax.set_ylim(0, 1)
            
            y_pos_recreate = 0.98
            for item in legend_items:
                vis_key = item['visibility_key']
                is_visible = visibility[vis_key]
                alpha_recreate = 0.3 if not is_visible else 1.0
                
                if item['symbol_type'] == 'line':
                    legend_ax.plot([0.02, 0.08], [y_pos_recreate, y_pos_recreate], 
                                 color=item['color'], linewidth=2, 
                                 linestyle='--' if 'Branch' in item['label'] else '-',
                                 alpha=alpha_recreate)
                elif item['symbol_type'] == 'scatter':
                    legend_ax.scatter([0.05], [y_pos_recreate], color=item['color'], 
                                    s=50 if 'CoM' in item['label'] or 'Body Points' in item['label'] or 'Body-leg' in item['label'] else 30, 
                                    marker=item['marker'],
                                    edgecolors='red' if 'CoM' in item['label'] else None, 
                                    linewidths=2 if 'CoM' in item['label'] else 0,
                                    alpha=alpha_recreate)
                elif item['symbol_type'] == 'surface':
                    # Draw filled rectangle for surface
                    legend_ax.add_patch(plt.Rectangle((0.02, y_pos_recreate - line_height/4), 0.06, line_height/2, 
                                                     facecolor=item['color'], alpha=0.2 * alpha_recreate, 
                                                     edgecolor=item['color'], linewidth=1))
                
                legend_ax.text(0.12, y_pos_recreate, item['label'], fontsize=9, 
                             verticalalignment='center', alpha=alpha_recreate,
                             color='gray' if not is_visible else 'black')
                
                # Recreate clickable rectangle
                rect_new = plt.Rectangle((0, y_pos_recreate - line_height/2), 0.22, line_height, 
                                        transform=legend_ax.transData, alpha=0, picker=True)
                legend_ax.add_patch(rect_new)
                item['rect'] = rect_new
                item['y'] = y_pos_recreate
                
                y_pos_recreate -= line_height * 1.5
        
        def on_legend_click(event):
            """Handle clicks on legend items."""
            if event.inaxes != legend_ax:
                return
            
            # Get click coordinates in legend axes
            x_click, y_click = event.xdata, event.ydata
            if x_click is None or y_click is None:
                return
            
            # Find which item was clicked
            for item in legend_items:
                y_item = item['y']
                # Check if click is within the item's vertical range
                if abs(y_click - y_item) < line_height:
                    # Toggle visibility
                    visibility_key = item['visibility_key']
                    visibility[visibility_key] = not visibility[visibility_key]
                    
                    # Redraw legend and plot
                    redraw_legend()
                    update_plot()
                    fig.canvas.draw()
                    break
        
        fig.canvas.mpl_connect('button_press_event', on_legend_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        update_plot()
        
        instructions = ('Controls:\n'
                       'Left/Right or A/D: Navigate frames\n'
                       'Up/Down or W/S: Switch between datasets\n'
                       'Click legend items to toggle visibility')
        plt.figtext(0.27, 0.02, instructions, 
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Ensure interactive mode is on
        plt.ion()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=False)


def main():
    root = tk.Tk()
    app = ProcessingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
