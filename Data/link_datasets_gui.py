"""
GUI tool to link ant datasets to calibration and branch sets.
"""

import json
import tkinter as tk
from tkinter import ttk, messagebox
import glob
from pathlib import Path

DATASET_LINKS_FILE = Path(__file__).parent / "dataset_links.json"
VIDEOS_2D_DATA_DIR = Path(__file__).parent / "Datasets" / "2D_data"
CALIBRATION_SETS_FILE = Path(__file__).parent / "Calibration" / "calibration_sets.json"
BRANCH_SETS_FILE = Path(__file__).parent / "Branch_Sets" / "branch_sets.json"


def load_dataset_links():
    """Load dataset links from JSON file."""
    if DATASET_LINKS_FILE.exists():
        try:
            with open(DATASET_LINKS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading dataset_links.json: {e}")
            return {}
    return {}


def save_dataset_links(dataset_links):
    """Save dataset links to JSON file."""
    DATASET_LINKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp_file = DATASET_LINKS_FILE.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(dataset_links, f, indent=2, ensure_ascii=False)
        if temp_file.exists():
            if DATASET_LINKS_FILE.exists():
                DATASET_LINKS_FILE.unlink()
            temp_file.replace(DATASET_LINKS_FILE)
    except Exception as e:
        print(f"Warning: Could not use atomic write, using direct write: {e}")
        with open(DATASET_LINKS_FILE, 'w') as f:
            json.dump(dataset_links, f, indent=2, ensure_ascii=False)


def load_calibration_sets():
    """Load calibration sets."""
    if CALIBRATION_SETS_FILE.exists():
        with open(CALIBRATION_SETS_FILE, 'r') as f:
            return json.load(f)
    return {}


def load_branch_sets():
    """Load branch sets."""
    if BRANCH_SETS_FILE.exists():
        with open(BRANCH_SETS_FILE, 'r') as f:
            return json.load(f)
    return {}


def find_ant_datasets():
    """Find all available ant datasets from 2D data files."""
    if not VIDEOS_2D_DATA_DIR.exists():
        return []
    
    # Find all files with camera suffixes (L, T, R, F)
    files = glob.glob(str(VIDEOS_2D_DATA_DIR / "*.xlsx"))
    
    # Extract unique dataset names (e.g., "11U1" from "11U1L.xlsx")
    datasets = set()
    for file in files:
        name = Path(file).stem
        # Remove camera suffix if present
        if name.endswith(('L', 'T', 'R', 'F')):
            dataset_name = name[:-1]
            datasets.add(dataset_name)
    
    return sorted(list(datasets))


def detect_species(dataset_name):
    """Detect species from dataset name."""
    if dataset_name.startswith(('11U', '12U')):
        return 'WR'
    elif dataset_name.startswith(('21U', '22U')):
        return 'NWR'
    return ''


class DatasetLinkManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Link Manager")
        self.root.geometry("1000x700")
        
        self.dataset_links = load_dataset_links()
        self.calibration_sets = load_calibration_sets()
        self.branch_sets = load_branch_sets()
        self.available_datasets = find_ant_datasets()
        
        # Main container
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        # Left panel - List of links
        left_frame = ttk.LabelFrame(main_frame, text="Dataset Links", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.links_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=20)
        self.links_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.links_listbox.yview)
        
        self.links_listbox.bind('<<ListboxSelect>>', self.on_link_select)
        
        # Buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="New", command=self.create_new).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Delete", command=self.delete_link).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Refresh", command=self.refresh_list).pack(side=tk.LEFT)
        
        # Right panel - Link Editor
        right_frame = ttk.LabelFrame(main_frame, text="Link Details", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Dataset selector
        dataset_frame = ttk.Frame(right_frame)
        dataset_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(dataset_frame, text="Dataset:").pack(side=tk.LEFT)
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(dataset_frame, textvariable=self.dataset_var, width=30, state="readonly")
        self.dataset_combo.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        self.dataset_combo.bind('<<ComboboxSelected>>', self.on_dataset_select)
        
        # Calibration set selector
        cal_frame = ttk.Frame(right_frame)
        cal_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(cal_frame, text="Calibration Set:").pack(side=tk.LEFT)
        self.cal_var = tk.StringVar()
        self.cal_combo = ttk.Combobox(cal_frame, textvariable=self.cal_var, width=27, state="readonly")
        self.cal_combo.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        self.cal_var.trace('w', lambda *args: self.update_info())
        
        # Branch set selector
        branch_frame = ttk.Frame(right_frame)
        branch_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(branch_frame, text="Branch Set:").pack(side=tk.LEFT)
        self.branch_var = tk.StringVar()
        self.branch_combo = ttk.Combobox(branch_frame, textvariable=self.branch_var, width=27, state="readonly")
        self.branch_combo.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        self.branch_var.trace('w', lambda *args: self.update_info())
        
        # Species selector
        species_frame = ttk.Frame(right_frame)
        species_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(species_frame, text="Species:").pack(side=tk.LEFT)
        self.species_var = tk.StringVar()
        self.species_combo = ttk.Combobox(species_frame, textvariable=self.species_var, 
                                          values=['WR', 'NWR'], width=27, state="readonly")
        self.species_combo.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        self.species_var.trace('w', lambda *args: self.update_info())
        
        # Info text
        info_frame = ttk.LabelFrame(right_frame, text="Information", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.info_text = tk.Text(info_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Save button
        save_btn = ttk.Button(right_frame, text="Save Link", command=self.save_link)
        save_btn.pack(pady=(10, 0))
        
        # Track editing state
        self.editing_dataset = None
        
        # Initialize
        self.refresh_calibration_list()
        self.refresh_branch_list()
        self.refresh_dataset_list()
        self.refresh_list()
        self.create_new()
    
    def refresh_calibration_list(self):
        """Refresh calibration sets list."""
        self.calibration_sets = load_calibration_sets()
        cal_names = sorted(self.calibration_sets.keys())
        self.cal_combo['values'] = cal_names
    
    def refresh_branch_list(self):
        """Refresh branch sets list."""
        self.branch_sets = load_branch_sets()
        branch_names = sorted(self.branch_sets.keys())
        self.branch_combo['values'] = branch_names
    
    def refresh_dataset_list(self):
        """Refresh available datasets list."""
        self.available_datasets = find_ant_datasets()
        # Update dropdown based on current mode (new vs edit)
        self._update_dataset_dropdown()
    
    def _update_dataset_dropdown(self):
        """Update dataset dropdown to show only unlinked datasets (or current dataset if editing)."""
        if hasattr(self, 'editing_dataset') and self.editing_dataset:
            # Editing mode: show current dataset + unlinked datasets
            unlinked = [d for d in self.available_datasets if d not in self.dataset_links]
            if self.editing_dataset not in unlinked:
                # Add current dataset if it's not already in the list
                values = [self.editing_dataset] + unlinked
            else:
                values = unlinked
            self.dataset_combo['values'] = sorted(values)
        else:
            # New mode: show only unlinked datasets
            unlinked = [d for d in self.available_datasets if d not in self.dataset_links]
            self.dataset_combo['values'] = sorted(unlinked)
    
    def refresh_list(self):
        """Refresh the list of links."""
        self.dataset_links = load_dataset_links()
        self.links_listbox.delete(0, tk.END)
        for name in sorted(self.dataset_links.keys()):
            link = self.dataset_links[name]
            cal = link.get('calibration_set', 'N/A')
            branch = link.get('branch_set', 'N/A')
            species = link.get('species', 'N/A')
            display = f"{name} | {cal} | {branch} | {species}"
            self.links_listbox.insert(tk.END, display)
        
        # Update dataset dropdown after refresh
        if hasattr(self, 'dataset_combo'):
            self._update_dataset_dropdown()
    
    def on_link_select(self, event):
        """Handle selection of a link."""
        selection = self.links_listbox.curselection()
        if not selection:
            return
        
        display_text = self.links_listbox.get(selection[0])
        dataset_name = display_text.split(' | ')[0]
        self.load_link(dataset_name)
    
    def load_link(self, dataset_name):
        """Load a link into the editor."""
        if dataset_name not in self.dataset_links:
            return
        
        # Set editing mode
        self.editing_dataset = dataset_name
        
        link = self.dataset_links[dataset_name]
        self.dataset_var.set(dataset_name)
        self.cal_var.set(link.get('calibration_set', ''))
        self.branch_var.set(link.get('branch_set', ''))
        self.species_var.set(link.get('species', ''))
        
        # Update dropdown to include current dataset
        self._update_dataset_dropdown()
        self.update_info()
    
    def on_dataset_select(self, event=None):
        """Handle dataset selection - auto-detect species."""
        dataset_name = self.dataset_var.get()
        if dataset_name:
            species = detect_species(dataset_name)
            if species:
                self.species_var.set(species)
            self.update_info()
    
    def create_new(self):
        """Create a new link."""
        # Clear editing mode
        self.editing_dataset = None
        
        self.dataset_var.set('')
        self.cal_var.set('')
        self.branch_var.set('')
        self.species_var.set('')
        self.links_listbox.selection_clear(0, tk.END)
        
        # Update dropdown to show only unlinked datasets
        self._update_dataset_dropdown()
        self.update_info()
    
    def update_info(self):
        """Update information text."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        
        dataset_name = self.dataset_var.get()
        if dataset_name:
            # Check if dataset exists
            if dataset_name in self.available_datasets:
                self.info_text.insert(tk.END, f"✓ Dataset '{dataset_name}' found in 2D_data folder\n\n")
            else:
                self.info_text.insert(tk.END, f"⚠ Dataset '{dataset_name}' not found in 2D_data folder\n\n")
            
            # Check calibration set
            cal_name = self.cal_var.get()
            if cal_name:
                if cal_name in self.calibration_sets:
                    self.info_text.insert(tk.END, f"✓ Calibration set '{cal_name}' found\n")
                else:
                    self.info_text.insert(tk.END, f"⚠ Calibration set '{cal_name}' not found\n")
            else:
                self.info_text.insert(tk.END, "⚠ No calibration set selected\n")
            
            # Check branch set
            branch_name = self.branch_var.get()
            if branch_name:
                if branch_name in self.branch_sets:
                    self.info_text.insert(tk.END, f"✓ Branch set '{branch_name}' found\n")
                else:
                    self.info_text.insert(tk.END, f"⚠ Branch set '{branch_name}' not found\n")
            else:
                self.info_text.insert(tk.END, "⚠ No branch set selected\n")
            
            # Species
            species = self.species_var.get()
            if species:
                self.info_text.insert(tk.END, f"✓ Species: {species}\n")
            else:
                self.info_text.insert(tk.END, "⚠ No species selected\n")
        else:
            self.info_text.insert(tk.END, "Select a dataset to create or edit a link.\n\n")
            self.info_text.insert(tk.END, f"Available datasets: {len(self.available_datasets)}\n")
            self.info_text.insert(tk.END, f"Calibration sets: {len(self.calibration_sets)}\n")
            self.info_text.insert(tk.END, f"Branch sets: {len(self.branch_sets)}\n")
            self.info_text.insert(tk.END, f"Existing links: {len(self.dataset_links)}\n")
        
        self.info_text.config(state=tk.DISABLED)
    
    def save_link(self):
        """Save the current link."""
        dataset_name = self.dataset_var.get().strip()
        if not dataset_name:
            messagebox.showerror("Error", "Please select a dataset.")
            return
        
        if dataset_name not in self.available_datasets:
            if not messagebox.askyesno("Warning", 
                    f"Dataset '{dataset_name}' not found in 2D_data folder.\n\n"
                    f"Continue anyway?"):
                return
        
        cal_name = self.cal_var.get().strip()
        if not cal_name:
            messagebox.showerror("Error", "Please select a calibration set.")
            return
        
        if cal_name not in self.calibration_sets:
            messagebox.showerror("Error", f"Calibration set '{cal_name}' not found.")
            return
        
        branch_name = self.branch_var.get().strip()
        if not branch_name:
            messagebox.showerror("Error", "Please select a branch set.")
            return
        
        if branch_name not in self.branch_sets:
            messagebox.showerror("Error", f"Branch set '{branch_name}' not found.")
            return
        
        species = self.species_var.get().strip()
        if not species:
            messagebox.showerror("Error", "Please select a species.")
            return
        
        if species not in ['WR', 'NWR']:
            messagebox.showerror("Error", "Species must be WR or NWR.")
            return
        
        # Save link
        self.dataset_links[dataset_name] = {
            "calibration_set": cal_name,
            "branch_set": branch_name,
            "species": species
        }
        
        save_dataset_links(self.dataset_links)
        self.refresh_list()
        
        # Update editing state
        self.editing_dataset = dataset_name
        
        # Select the saved link
        items = self.links_listbox.get(0, tk.END)
        for idx, item in enumerate(items):
            if item.startswith(dataset_name + ' | '):
                self.links_listbox.selection_set(idx)
                self.links_listbox.see(idx)
                break
        
        # Update dropdown to reflect new link status
        self._update_dataset_dropdown()
        
        messagebox.showinfo("Success", f"Link saved for dataset '{dataset_name}'!")
        self.update_info()
    
    def delete_link(self):
        """Delete the selected link."""
        selection = self.links_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a link to delete.")
            return
        
        display_text = self.links_listbox.get(selection[0])
        dataset_name = display_text.split(' | ')[0]
        
        if messagebox.askyesno("Confirm", f"Delete link for dataset '{dataset_name}'?"):
            if dataset_name in self.dataset_links:
                del self.dataset_links[dataset_name]
                save_dataset_links(self.dataset_links)
                self.refresh_list()
                self.create_new()  # This will clear editing state and update dropdown
                messagebox.showinfo("Success", f"Link deleted for '{dataset_name}'.")


def main():
    root = tk.Tk()
    app = DatasetLinkManager(root)
    root.mainloop()


if __name__ == "__main__":
    main()
