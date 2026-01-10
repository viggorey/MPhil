"""
Unified GUI for Ant Tracking Analysis Workflow.
Combines all steps: Calibration, Branch Sets, Dataset Linking, 
3D Reconstruction, Parameterization, Analysis, and Figures.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import sys


class EmbeddedRoot:
    """Adapter class to make a Frame act like a Tk root for embedded GUIs."""
    def __init__(self, frame, master_root):
        self.frame = frame
        self.master_root = master_root
        self._title = ""
        self._geometry = ""
    
    def title(self, title=None):
        if title is not None:
            self._title = title
            self.master_root.title(f"Ant Tracking Analysis Workflow - {title}")
        return self._title
    
    def geometry(self, geometry=None):
        if geometry is not None:
            self._geometry = geometry
            # Don't change master window geometry when embedded
        return self._geometry
    
    def columnconfigure(self, *args, **kwargs):
        return self.frame.columnconfigure(*args, **kwargs)
    
    def rowconfigure(self, *args, **kwargs):
        return self.frame.rowconfigure(*args, **kwargs)
    
    def __getattr__(self, name):
        # Delegate all other attributes to the frame
        return getattr(self.frame, name)

# Add paths for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "Processing"))
sys.path.insert(0, str(BASE_DIR / "Data" / "Calibration"))
sys.path.insert(0, str(BASE_DIR / "Data" / "Videos"))
sys.path.insert(0, str(BASE_DIR / "Data"))

# Import individual GUI modules
# Note: We import by adding paths and using direct imports
import importlib.util

# Import calibration manager
calibration_path = BASE_DIR / "Data" / "Calibration" / "manage_calibration_sets_gui.py"
spec = importlib.util.spec_from_file_location("manage_calibration_sets_gui", calibration_path)
calibration_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration_module)
CalibrationSetManager = calibration_module.CalibrationSetManager

# Import branch set manager
branch_path = BASE_DIR / "Data" / "Videos" / "manage_branch_sets_gui.py"
spec = importlib.util.spec_from_file_location("manage_branch_sets_gui", branch_path)
branch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(branch_module)
BranchSetManager = branch_module.BranchSetManager

# Import dataset link manager
dataset_path = BASE_DIR / "Data" / "link_datasets_gui.py"
spec = importlib.util.spec_from_file_location("link_datasets_gui", dataset_path)
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)
DatasetLinkManager = dataset_module.DatasetLinkManager

# Import processing GUI
processing_path = BASE_DIR / "Processing" / "master_processing_gui.py"
spec = importlib.util.spec_from_file_location("master_processing_gui", processing_path)
processing_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(processing_module)
ProcessingGUI = processing_module.ProcessingGUI

# Import analysis/figures GUI
analysis_path = BASE_DIR / "Processing" / "master_analysis_figures_gui.py"
spec = importlib.util.spec_from_file_location("master_analysis_figures_gui", analysis_path)
analysis_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis_module)
AnalysisFiguresGUI = analysis_module.AnalysisFiguresGUI


class MasterWorkflowGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ant Tracking Analysis Workflow")
        self.root.geometry("1400x900")
        
        # Current active module
        self.current_module = None
        self.current_frame = None
        
        # Create main layout
        self._create_layout()
        
        # Load default module (Calibration)
        self.show_module("calibration")
    
    def _create_layout(self):
        """Create the main layout with sidebar and content area."""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Sidebar
        sidebar = ttk.Frame(main_container, width=200)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        sidebar.pack_propagate(False)
        
        # Sidebar title
        title_label = ttk.Label(sidebar, text="Workflow Steps", font=("Arial", 12, "bold"))
        title_label.pack(pady=(10, 20))
        
        # Export/Import buttons
        export_import_frame = ttk.LabelFrame(sidebar, text="Data Management", padding="10")
        export_import_frame.pack(pady=(20, 0), padx=10, fill=tk.X)
        
        ttk.Button(export_import_frame, text="Export All Data", 
                  command=self._export_data, width=18).pack(pady=5, fill=tk.X)
        ttk.Button(export_import_frame, text="Import All Data", 
                  command=self._import_data, width=18).pack(pady=5, fill=tk.X)
        
        # Navigation buttons
        self.nav_buttons = {}
        nav_items = [
            ("1. Calibration", "calibration"),
            ("2. Branch Sets", "branch_sets"),
            ("3. Datasets", "datasets"),
            ("4. 3D Reconstruction", "reconstruction"),
            ("5. Parameterization", "parameterization"),
            ("6. Analysis", "analysis"),
            ("7. Figures", "figures")
        ]
        
        for text, key in nav_items:
            btn = ttk.Button(
                sidebar, 
                text=text, 
                command=lambda k=key: self.show_module(k),
                width=20
            )
            btn.pack(pady=5, padx=10, fill=tk.X)
            self.nav_buttons[key] = btn
        
        # Content area
        self.content_frame = ttk.Frame(main_container)
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    def show_module(self, module_key):
        """Show the selected module in the content area."""
        # Update button states
        for key, btn in self.nav_buttons.items():
            if key == module_key:
                btn.state(['pressed'])
            else:
                btn.state(['!pressed'])
        
        # Clear current content
        if self.current_frame:
            self.current_frame.destroy()
            self.current_module = None
        
        # Create container frame in content area
        container = ttk.Frame(self.content_frame)
        container.pack(fill=tk.BOTH, expand=True)
        self.current_frame = container
        
        # Create new module frame
        if module_key == "calibration":
            self._show_calibration(container)
        elif module_key == "branch_sets":
            self._show_branch_sets(container)
        elif module_key == "datasets":
            self._show_datasets(container)
        elif module_key == "reconstruction":
            self._show_reconstruction(container)
        elif module_key == "parameterization":
            self._show_parameterization(container)
        elif module_key == "analysis":
            self._show_analysis(container)
        elif module_key == "figures":
            self._show_figures(container)
    
    def _show_calibration(self, container):
        """Show calibration set manager."""
        # Create embedded frame
        embedded_frame = tk.Frame(container)
        embedded_frame.pack(fill=tk.BOTH, expand=True)
        embedded_frame.columnconfigure(0, weight=1)
        embedded_frame.rowconfigure(0, weight=1)
        
        # Create adapter to make frame act like root
        embedded_root = EmbeddedRoot(embedded_frame, self.root)
        
        # Create calibration manager
        calibration_manager = CalibrationSetManager(embedded_root)
        self.current_module = calibration_manager
    
    def _show_branch_sets(self, container):
        """Show branch set manager."""
        embedded_frame = tk.Frame(container)
        embedded_frame.pack(fill=tk.BOTH, expand=True)
        embedded_frame.columnconfigure(0, weight=1)
        embedded_frame.rowconfigure(0, weight=1)
        
        embedded_root = EmbeddedRoot(embedded_frame, self.root)
        branch_manager = BranchSetManager(embedded_root)
        self.current_module = branch_manager
    
    def _show_datasets(self, container):
        """Show dataset link manager."""
        embedded_frame = tk.Frame(container)
        embedded_frame.pack(fill=tk.BOTH, expand=True)
        embedded_frame.columnconfigure(0, weight=1)
        embedded_frame.rowconfigure(0, weight=1)
        
        embedded_root = EmbeddedRoot(embedded_frame, self.root)
        dataset_manager = DatasetLinkManager(embedded_root)
        self.current_module = dataset_manager
    
    def _show_reconstruction(self, container):
        """Show 3D reconstruction interface."""
        embedded_frame = tk.Frame(container)
        embedded_frame.pack(fill=tk.BOTH, expand=True)
        embedded_frame.columnconfigure(0, weight=1)
        embedded_frame.rowconfigure(0, weight=1)
        
        embedded_root = EmbeddedRoot(embedded_frame, self.root)
        processing_gui = ProcessingGUI(embedded_root, embedded_mode=True, default_tab=0)
        self.current_module = processing_gui
    
    def _show_parameterization(self, container):
        """Show parameterization interface."""
        embedded_frame = tk.Frame(container)
        embedded_frame.pack(fill=tk.BOTH, expand=True)
        embedded_frame.columnconfigure(0, weight=1)
        embedded_frame.rowconfigure(0, weight=1)
        
        embedded_root = EmbeddedRoot(embedded_frame, self.root)
        processing_gui = ProcessingGUI(embedded_root, embedded_mode=True, default_tab=1)
        self.current_module = processing_gui
    
    def _show_analysis(self, container):
        """Show statistical analysis interface."""
        embedded_frame = tk.Frame(container)
        embedded_frame.pack(fill=tk.BOTH, expand=True)
        embedded_frame.columnconfigure(0, weight=1)
        embedded_frame.rowconfigure(0, weight=1)
        
        embedded_root = EmbeddedRoot(embedded_frame, self.root)
        analysis_gui = AnalysisFiguresGUI(embedded_root, embedded_mode=True, default_tab=0)
        self.current_module = analysis_gui
    
    def _show_figures(self, container):
        """Show figure generation interface."""
        embedded_frame = tk.Frame(container)
        embedded_frame.pack(fill=tk.BOTH, expand=True)
        embedded_frame.columnconfigure(0, weight=1)
        embedded_frame.rowconfigure(0, weight=1)
        
        embedded_root = EmbeddedRoot(embedded_frame, self.root)
        figures_gui = AnalysisFiguresGUI(embedded_root, embedded_mode=True, default_tab=1)
        self.current_module = figures_gui
    
    def _export_data(self):
        """Export all data to a master JSON file."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from export_import import export_all_data
        
        # Ask for save location
        output_file = filedialog.asksaveasfilename(
            title="Export All Data",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not output_file:
            return
        
        try:
            # Show progress
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Exporting Data...")
            progress_window.geometry("400x100")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="Exporting all data...")
            progress_label.pack(pady=20)
            
            self.root.update()
            
            # Export data
            exported_file = export_all_data(output_file)
            
            progress_window.destroy()
            
            messagebox.showinfo(
                "Export Complete",
                f"All data exported successfully!\n\n"
                f"File: {exported_file}\n\n"
                f"Exported:\n"
                f"- Calibration sets\n"
                f"- Branch sets\n"
                f"- Dataset links\n"
                f"- 2D data files\n"
                f"- 3D data files\n"
                f"- Parameterized data files"
            )
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("Export Error", f"Error exporting data:\n{e}")
    
    def _import_data(self):
        """Import all data from a master JSON file."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from export_import import import_all_data
        
        # Ask for file to import
        input_file = filedialog.askopenfilename(
            title="Import All Data",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not input_file:
            return
        
        # Ask about overwrite
        overwrite = messagebox.askyesno(
            "Import Options",
            "Do you want to overwrite existing files?\n\n"
            "Yes: Overwrite existing files\n"
            "No: Skip existing files (only import new data)"
        )
        
        try:
            # Show progress
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Importing Data...")
            progress_window.geometry("400x100")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="Importing all data...")
            progress_label.pack(pady=20)
            
            self.root.update()
            
            # Import data
            summary = import_all_data(input_file, overwrite=overwrite)
            
            progress_window.destroy()
            
            # Show summary
            summary_text = "Import Complete!\n\n"
            
            # Check if anything was imported or skipped
            total_imported = (
                summary['calibration_sets'].get('imported', summary['calibration_sets'] if isinstance(summary['calibration_sets'], int) else 0) +
                summary['branch_sets'].get('imported', summary['branch_sets'] if isinstance(summary['branch_sets'], int) else 0) +
                summary['dataset_links'].get('imported', summary['dataset_links'] if isinstance(summary['dataset_links'], int) else 0) +
                summary['2d_data'].get('imported', summary['2d_data'] if isinstance(summary['2d_data'], int) else 0) +
                summary['3d_data'].get('imported', summary['3d_data'] if isinstance(summary['3d_data'], int) else 0) +
                summary['parameterized_data'].get('imported', summary['parameterized_data'] if isinstance(summary['parameterized_data'], int) else 0)
            )
            
            total_skipped = (
                summary['calibration_sets'].get('skipped', 0) +
                summary['branch_sets'].get('skipped', 0) +
                summary['dataset_links'].get('skipped', 0) +
                summary['2d_data'].get('skipped', 0) +
                summary['3d_data'].get('skipped', 0) +
                summary['parameterized_data'].get('skipped', 0)
            )
            
            if isinstance(summary['calibration_sets'], dict):
                # New format with imported/skipped
                summary_text += "Imported:\n"
                summary_text += f"- Calibration sets: {summary['calibration_sets']['imported']}\n"
                summary_text += f"- Branch sets: {summary['branch_sets']['imported']}\n"
                summary_text += f"- Dataset links: {summary['dataset_links']['imported']}\n"
                summary_text += f"- 2D data files: {summary['2d_data']['imported']}\n"
                summary_text += f"- 3D data files: {summary['3d_data']['imported']}\n"
                summary_text += f"- Parameterized data files: {summary['parameterized_data']['imported']}\n"
                
                if total_skipped > 0:
                    summary_text += f"\nSkipped (files already exist):\n"
                    summary_text += f"- Calibration sets: {summary['calibration_sets']['skipped']}\n"
                    summary_text += f"- Branch sets: {summary['branch_sets']['skipped']}\n"
                    summary_text += f"- Dataset links: {summary['dataset_links']['skipped']}\n"
                    summary_text += f"- 2D data files: {summary['2d_data']['skipped']}\n"
                    summary_text += f"- 3D data files: {summary['3d_data']['skipped']}\n"
                    summary_text += f"- Parameterized data files: {summary['parameterized_data']['skipped']}\n"
                    summary_text += f"\nNote: To overwrite existing files, select 'Yes' when asked about overwriting."
            else:
                # Old format (backward compatibility)
                summary_text += "Imported:\n"
                summary_text += f"- Calibration sets: {summary['calibration_sets']}\n"
                summary_text += f"- Branch sets: {summary['branch_sets']}\n"
                summary_text += f"- Dataset links: {summary['dataset_links']}\n"
                summary_text += f"- 2D data files: {summary['2d_data']}\n"
                summary_text += f"- 3D data files: {summary['3d_data']}\n"
                summary_text += f"- Parameterized data files: {summary['parameterized_data']}\n"
            
            if summary['errors']:
                summary_text += f"\nErrors ({len(summary['errors'])}):\n"
                for error in summary['errors'][:5]:  # Show first 5 errors
                    summary_text += f"  - {error}\n"
                if len(summary['errors']) > 5:
                    summary_text += f"  ... and {len(summary['errors']) - 5} more\n"
            
            messagebox.showinfo("Import Complete", summary_text)
            
            # Refresh current module if it exists
            if self.current_module:
                # Try to refresh the current module
                if hasattr(self.current_module, 'refresh_list'):
                    self.current_module.refresh_list()
                elif hasattr(self.current_module, 'refresh'):
                    self.current_module.refresh()
            
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("Import Error", f"Error importing data:\n{e}")


def main():
    root = tk.Tk()
    app = MasterWorkflowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
