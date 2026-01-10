"""
GUI for Statistical Analysis and Figure Generation.
Allows selecting metrics for analysis and figures to generate.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import json
import threading
from pathlib import Path
import sys
from datetime import datetime
import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import webbrowser

# Add paths for imports
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = Path(__file__).parent.parent / "Analysis" / "results"
FIGURES_DIR = Path(__file__).parent.parent / "Analysis" / "figures"
PARAM_DIR = BASE_DIR / "Data" / "Datasets" / "3D_data_params"

# Import analysis and figure functions
sys.path.insert(0, str(Path(__file__).parent))
from master_analysis import ALL_METRICS, analyze_single_metric
from master_figures import (
    create_figure_1_speed_stride, create_figure_2_com_distance, 
    create_figure_3_pull_off_force, FIGURES_DIR as FIGURES_DIR_MODULE
)

# Ensure directories exist
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


class AnalysisFiguresGUI:
    def __init__(self, root, embedded_mode=False, default_tab=0):
        self.root = root
        self.embedded_mode = embedded_mode
        
        if not embedded_mode:
            self.root.title("Statistical Analysis & Figure Generation")
            self.root.geometry("1200x800")
        
        if embedded_mode:
            # In embedded mode, just create the requested tab directly without notebook
            if default_tab == 0:
                self.analysis_frame = ttk.Frame(root)
                self.analysis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                self._setup_analysis_tab()
                self.notebook = None  # No notebook in embedded mode
            else:
                self.figures_frame = ttk.Frame(root)
                self.figures_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                self._setup_figures_tab()
                self.notebook = None  # No notebook in embedded mode
        else:
            # Create notebook for tabs (only when not embedded)
            self.notebook = ttk.Notebook(root)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Analysis tab
            self.analysis_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.analysis_frame, text="Statistical Analysis")
            self._setup_analysis_tab()
            
            # Figures tab
            self.figures_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.figures_frame, text="Figure Generation")
            self._setup_figures_tab()
        
        # Variables
        self.analysis_thread = None
        self.figures_thread = None
        self.is_analyzing = False
        self.is_generating_figures = False
        self.last_generated_figure = None  # Track most recently generated figure
        self.figure_preview_map = {}  # Map display names to file paths for preview
        
    def _setup_analysis_tab(self):
        """Setup the Analysis tab."""
        # Top frame with buttons
        top_frame = ttk.Frame(self.analysis_frame)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(top_frame, text="Select All", command=self._select_all_metrics).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="Deselect All", command=self._deselect_all_metrics).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="Run Analysis", command=self._run_analysis).pack(side=tk.LEFT, padx=10)
        ttk.Button(top_frame, text="Refresh", command=self._refresh_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="Open Results Directory", command=self._open_analysis_dir).pack(side=tk.LEFT, padx=10)
        ttk.Button(top_frame, text="View Analysis Results", command=self._view_analysis_results).pack(side=tk.LEFT, padx=2)
        
        # Main content frame with scrollable list
        main_frame = ttk.Frame(self.analysis_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create scrollable frame for metrics
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.metrics_frame = ttk.Frame(canvas)
        
        self.metrics_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.metrics_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind mouse wheel scrolling (cross-platform)
        def _on_mousewheel(event):
            if sys.platform == "win32":
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif sys.platform == "darwin":
                canvas.yview_scroll(int(-1 * event.delta), "units")
            else:
                # Linux
                if event.num == 4:
                    canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    canvas.yview_scroll(1, "units")
        
        def _bind_to_mousewheel(event):
            if sys.platform == "win32":
                canvas.bind_all("<MouseWheel>", _on_mousewheel)
            elif sys.platform == "darwin":
                canvas.bind_all("<MouseWheel>", _on_mousewheel)
            else:
                # Linux
                canvas.bind_all("<Button-4>", _on_mousewheel)
                canvas.bind_all("<Button-5>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            if sys.platform == "win32":
                canvas.unbind_all("<MouseWheel>")
            elif sys.platform == "darwin":
                canvas.unbind_all("<MouseWheel>")
            else:
                # Linux
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create checkboxes for all metrics with status indicators
        self.metric_vars = {}
        self.metric_widgets = {}  # Store widget references for click handling
        for metric in ALL_METRICS:
            var = tk.BooleanVar(value=True)
            self.metric_vars[metric] = var
            
            # Check if this metric has been analyzed
            metric_prefix, parameter_name = metric.split(":")
            from master_analysis import map_metric_to_sheet
            sheet_name = map_metric_to_sheet(metric_prefix)
            results_dir = ANALYSIS_DIR / metric_prefix / parameter_name
            is_analyzed = (results_dir / "analysis_results.txt").exists()
            
            # Create frame for each metric with checkbox and status
            metric_frame = ttk.Frame(self.metrics_frame)
            metric_frame.pack(fill=tk.X, padx=5, pady=2)
            
            cb = ttk.Checkbutton(metric_frame, text=metric, variable=var)
            cb.pack(side=tk.LEFT, anchor=tk.W)
            
            # Add status indicator
            if is_analyzed:
                status_label = ttk.Label(metric_frame, text="✓ Analyzed", foreground="green", cursor="hand2")
                status_label.pack(side=tk.LEFT, padx=(10, 0))
                # Make clickable to view results
                status_label.bind("<Button-1>", lambda e, m=metric: self._view_metric_results(m))
                cb.bind("<Button-1>", lambda e, m=metric: self._view_metric_results(m) if e.state & 0x4 else None)  # Double-click
            else:
                status_label = ttk.Label(metric_frame, text="Not analyzed", foreground="gray")
                status_label.pack(side=tk.LEFT, padx=(10, 0))
            
            self.metric_widgets[metric] = {
                'frame': metric_frame,
                'checkbox': cb,
                'status': status_label,
                'analyzed': is_analyzed
            }
        
        # Progress and log frame
        progress_frame = ttk.Frame(self.analysis_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.analysis_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.analysis_progress.pack(fill=tk.X, pady=2)
        
        self.analysis_log = scrolledtext.ScrolledText(progress_frame, height=10, wrap=tk.WORD)
        self.analysis_log.pack(fill=tk.BOTH, expand=True)
        
    def _setup_figures_tab(self):
        """Setup the Figures tab."""
        # Top frame with buttons
        top_frame = ttk.Frame(self.figures_frame)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(top_frame, text="Select All", command=self._select_all_figures).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="Deselect All", command=self._deselect_all_figures).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="Generate Figures", command=self._generate_figures).pack(side=tk.LEFT, padx=10)
        ttk.Button(top_frame, text="Refresh", command=self._refresh_figures).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="Open Figures Directory", command=self._open_figures_dir).pack(side=tk.LEFT, padx=10)
        
        # Main content frame
        main_frame = ttk.Frame(self.figures_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left side: Figure selection
        left_frame = ttk.LabelFrame(main_frame, text="Select Figures to Generate")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Create scrollable frame for figure checkboxes
        canvas = tk.Canvas(left_frame)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        self.figures_list_frame = ttk.Frame(canvas)
        
        self.figures_list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.figures_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Boxplot Generator (moved to top)
        ttk.Label(self.figures_list_frame, text="Boxplot Generator:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5, pady=(5, 2))
        
        # Checkbox for "Generate all in one figure" - moved to top for visibility
        self.multi_param_var = tk.BooleanVar(value=True)  # Default to True (generate all in one)
        ttk.Checkbutton(self.figures_list_frame, text="Generate all in one figure", variable=self.multi_param_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Create scrollable frame for multi-parameter metrics
        # Create container first
        multi_container = ttk.Frame(self.figures_list_frame)
        multi_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        
        # Create canvas and scrollbar inside container
        multi_canvas = tk.Canvas(multi_container, height=200)
        multi_scrollbar = ttk.Scrollbar(multi_container, orient="vertical", command=multi_canvas.yview)
        
        # Create frame for metrics inside canvas
        self.multi_metrics_frame = ttk.Frame(multi_canvas)
        
        self.multi_metrics_frame.bind(
            "<Configure>",
            lambda e: multi_canvas.configure(scrollregion=multi_canvas.bbox("all"))
        )
        
        multi_canvas.create_window((0, 0), window=self.multi_metrics_frame, anchor="nw")
        multi_canvas.configure(yscrollcommand=multi_scrollbar.set)
        
        # Create checkboxes for metrics
        self.multi_metric_vars = {}
        for metric in ALL_METRICS:
            var = tk.BooleanVar(value=False)
            self.multi_metric_vars[metric] = var
            cb = ttk.Checkbutton(self.multi_metrics_frame, text=metric, variable=var)
            cb.pack(anchor=tk.W, padx=5, pady=1)
        
        # Pack canvas and scrollbar into container
        multi_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        multi_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Gait Figure
        ttk.Separator(self.figures_list_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=10)
        ttk.Label(self.figures_list_frame, text="Gait Pattern Figure:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5, pady=(5, 2))
        self.gait_figure_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.figures_list_frame, text="Gait Pattern Heatmap", variable=self.gait_figure_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Boxplot by Feet Attached
        ttk.Separator(self.figures_list_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=10)
        ttk.Label(self.figures_list_frame, text="Boxplot by Feet Attached:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5, pady=(5, 2))
        
        feet_frame = ttk.Frame(self.figures_list_frame)
        feet_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(feet_frame, text="Metric:").pack(side=tk.LEFT, padx=2)
        self.feet_metric_var = tk.StringVar()
        self.feet_metric_combo = ttk.Combobox(feet_frame, textvariable=self.feet_metric_var, 
                                             state="readonly", width=15)  # Made much thinner (was 40)
        self.feet_metric_combo['values'] = ALL_METRICS
        self.feet_metric_combo.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(feet_frame, text="Feet:").pack(side=tk.LEFT, padx=(10, 2))
        self.num_feet_var = tk.StringVar(value="3")
        num_feet_spin = ttk.Spinbox(feet_frame, from_=0, to=6, textvariable=self.num_feet_var, width=5)
        num_feet_spin.pack(side=tk.LEFT, padx=2)
        
        self.feet_boxplot_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.figures_list_frame, text="Generate Boxplot", variable=self.feet_boxplot_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Right side: Preview area
        preview_frame = ttk.LabelFrame(main_frame, text="Figure Preview")
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Figure selection dropdown
        preview_top = ttk.Frame(preview_frame)
        preview_top.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(preview_top, text="Preview:").pack(side=tk.LEFT, padx=2)
        self.preview_var = tk.StringVar()
        self.preview_dropdown = ttk.Combobox(preview_top, textvariable=self.preview_var, 
                                             state="readonly", width=30)
        self.preview_dropdown['values'] = []  # Will be populated by _update_preview_dropdown
        self.preview_dropdown.pack(side=tk.LEFT, padx=2)
        self.preview_dropdown.bind("<<ComboboxSelected>>", self._preview_figure)
        
        # Preview canvas
        self.preview_canvas_frame = ttk.Frame(preview_frame)
        self.preview_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Progress and log frame
        progress_frame = ttk.Frame(self.figures_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.figures_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.figures_progress.pack(fill=tk.X, pady=2)
        
        self.figures_log = scrolledtext.ScrolledText(progress_frame, height=8, wrap=tk.WORD)
        self.figures_log.pack(fill=tk.BOTH, expand=True)
        
    def _select_all_metrics(self):
        """Select all metrics."""
        for var in self.metric_vars.values():
            var.set(True)
    
    def _deselect_all_metrics(self):
        """Deselect all metrics."""
        for var in self.metric_vars.values():
            var.set(False)
    
    def _select_all_figures(self):
        """Select all figures."""
        for var in self.figure_vars.values():
            var.set(True)
    
    def _deselect_all_figures(self):
        """Deselect all figures."""
        for var in self.figure_vars.values():
            var.set(False)
    
    def _refresh_analysis(self):
        """Refresh analysis tab and update status indicators."""
        # Preserve current checkbox states
        current_states = {metric: var.get() for metric, var in self.metric_vars.items()}
        
        # Rebuild the metrics list with updated status
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()
        
        self.metric_widgets = {}
        for metric in ALL_METRICS:
            # Preserve existing state or default to True
            if metric in self.metric_vars:
                var = self.metric_vars[metric]
            else:
                var = tk.BooleanVar(value=current_states.get(metric, True))
                self.metric_vars[metric] = var
            
            # Check if this metric has been analyzed
            metric_prefix, parameter_name = metric.split(":")
            from master_analysis import map_metric_to_sheet
            sheet_name = map_metric_to_sheet(metric_prefix)
            results_dir = ANALYSIS_DIR / metric_prefix / parameter_name
            is_analyzed = (results_dir / "analysis_results.txt").exists()
            
            # Create frame for each metric with checkbox and status
            metric_frame = ttk.Frame(self.metrics_frame)
            metric_frame.pack(fill=tk.X, padx=5, pady=2)
            
            cb = ttk.Checkbutton(metric_frame, text=metric, variable=var)
            cb.pack(side=tk.LEFT, anchor=tk.W)
            
            # Add status indicator
            if is_analyzed:
                status_label = ttk.Label(metric_frame, text="✓ Analyzed", foreground="green", cursor="hand2")
                status_label.pack(side=tk.LEFT, padx=(10, 0))
                # Make clickable to view results
                status_label.bind("<Button-1>", lambda e, m=metric: self._view_metric_results(m))
                cb.bind("<Double-Button-1>", lambda e, m=metric: self._view_metric_results(m))
            else:
                status_label = ttk.Label(metric_frame, text="Not analyzed", foreground="gray")
                status_label.pack(side=tk.LEFT, padx=(10, 0))
            
            self.metric_widgets[metric] = {
                'frame': metric_frame,
                'checkbox': cb,
                'status': status_label,
                'analyzed': is_analyzed
            }
        
        self._log_analysis("Refreshed analysis tab - status indicators updated\n")
    
    def _refresh_figures(self):
        """Refresh figures tab."""
        self._log_figures("Refreshed figures tab")
        # Update preview dropdown if new figures exist
        self._update_preview_dropdown()
    
    def _update_preview_dropdown(self):
        """Update preview dropdown with the 10 most recently generated figures."""
        figure_files_with_time = []
        
        # Collect all figure files with their modification times
        # Gait figure
        gait_file = FIGURES_DIR / "Gait_Pattern_Heatmap.png"
        if gait_file.exists():
            figure_files_with_time.append((
                gait_file.stat().st_mtime,
                "Gait Pattern Heatmap",
                gait_file
            ))
        
        # Feet-attached boxplots
        for fig_file in FIGURES_DIR.glob("*_feet_boxplot.png"):
            display_name = fig_file.stem.replace("_", " ")
            figure_files_with_time.append((
                fig_file.stat().st_mtime,
                display_name,
                fig_file
            ))
        
        # Multi-parameter boxplots
        for fig_file in FIGURES_DIR.glob("Multi_Parameter_Boxplot_*.png"):
            figure_files_with_time.append((
                fig_file.stat().st_mtime,
                "Multi-Parameter Boxplot",
                fig_file
            ))
        
        # Individual parameter boxplots
        for fig_file in FIGURES_DIR.glob("*_boxplot.png"):
            # Exclude feet boxplots and multi-parameter boxplots (already added)
            if "_feet_boxplot" not in fig_file.stem and "Multi_Parameter_Boxplot" not in fig_file.stem:
                param_name = fig_file.stem.replace("_boxplot", "")
                display_name = f"Boxplot: {param_name}"
                figure_files_with_time.append((
                    fig_file.stat().st_mtime,
                    display_name,
                    fig_file
                ))
        
        # Sort by modification time (most recent first) and limit to 10
        figure_files_with_time.sort(key=lambda x: x[0], reverse=True)
        recent_figures = figure_files_with_time[:10]
        
        # Extract display names for dropdown, making them unique if needed
        available = []
        name_counts = {}
        self.figure_preview_map = {}
        
        for mtime, name, file_path in recent_figures:
            # Make display name unique if there are duplicates
            if name in name_counts:
                name_counts[name] += 1
                display_name = f"{name} ({name_counts[name]})"
            else:
                name_counts[name] = 0
                display_name = name
            
            available.append(display_name)
            self.figure_preview_map[display_name] = file_path
        
        if available:
            self.preview_dropdown['values'] = available
    
    def _preview_most_recent(self):
        """Preview the most recently generated figure."""
        if not self.last_generated_figure or not self.last_generated_figure.exists():
            return
        
        # Clear previous preview
        for widget in self.preview_canvas_frame.winfo_children():
            widget.destroy()
        
        # Load and display image
        try:
            img = Image.open(self.last_generated_figure)
            # Resize to fit preview area
            max_width = 600
            max_height = 500
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            label = ttk.Label(self.preview_canvas_frame, image=photo)
            label.image = photo  # Keep a reference
            label.pack(padx=5, pady=5)
            
            # Update dropdown to show this figure
            fig_name = self.last_generated_figure.stem.replace("_", " ")
            if fig_name not in self.preview_dropdown['values']:
                current_values = list(self.preview_dropdown['values'])
                current_values.insert(0, fig_name)
                self.preview_dropdown['values'] = current_values
            self.preview_var.set(fig_name)
        except Exception as e:
            messagebox.showerror("Preview Error", f"Could not load figure: {e}")
    
    def _run_analysis(self):
        """Run analysis on selected metrics."""
        if self.is_analyzing:
            messagebox.showwarning("Analysis Running", "Analysis is already in progress!")
            return
        
        selected_metrics = [metric for metric, var in self.metric_vars.items() if var.get()]
        
        if not selected_metrics:
            messagebox.showwarning("No Selection", "Please select at least one metric to analyze.")
            return
        
        self.is_analyzing = True
        self.analysis_thread = threading.Thread(target=self._run_analysis_thread, args=(selected_metrics,), daemon=True)
        self.analysis_thread.start()
    
    def _run_analysis_thread(self, metrics):
        """Run analysis in background thread."""
        try:
            total = len(metrics)
            self.root.after(0, lambda: self.analysis_progress.config(maximum=total, value=0))
            self.root.after(0, lambda: self._log_analysis(f"Starting analysis of {total} metric(s)...\n"))
            
            successful_count = 0
            skipped_count = 0
            failed_count = 0
            
            for i, metric in enumerate(metrics, 1):
                self.root.after(0, lambda m=metric, idx=i: self._log_analysis(f"[{idx}/{total}] Analyzing: {m}"))
                
                try:
                    summary = analyze_single_metric(metric)
                    if summary:
                        method = summary.get('method', 'Unknown')
                        
                        # Check if analysis was actually completed (not skipped)
                        if method == 'Skipped':
                            skipped_count += 1
                            reason = summary.get('reason', 'Insufficient data')
                            self.root.after(0, lambda m=metric, r=reason: self._log_analysis(
                                f"  ⚠ Skipped: {r}\n"
                            ))
                        else:
                            # Verify that results file was actually created
                            metric_prefix, parameter_name = metric.split(":")
                            from master_analysis import map_metric_to_sheet
                            sheet_name = map_metric_to_sheet(metric_prefix)
                            results_file = ANALYSIS_DIR / metric_prefix / parameter_name / "analysis_results.txt"
                            
                            if results_file.exists():
                                successful_count += 1
                                self.root.after(0, lambda s=summary: self._log_analysis(
                                    f"  ✓ {s.get('method', 'Unknown')}: Species p={s.get('species_p', 'N/A'):.4e}, "
                                    f"Substrate p={s.get('substrate_p', 'N/A'):.4e}\n"
                                ))
                            else:
                                failed_count += 1
                                self.root.after(0, lambda m=metric: self._log_analysis(
                                    f"  ⚠ Warning: {m} - No results file was created. This may be due to insufficient data.\n"
                                ))
                except Exception as e:
                    failed_count += 1
                    self.root.after(0, lambda e=e, m=metric: self._log_analysis(f"  ✗ Error: {e}\n"))
                
                self.root.after(0, lambda idx=i: self.analysis_progress.config(value=idx))
            
            # Final summary
            summary_msg = f"\nAnalysis complete! Successful: {successful_count}, Skipped: {skipped_count}, Failed: {failed_count}\n"
            if skipped_count > 0 or failed_count > 0:
                summary_msg += f"⚠ Warning: {skipped_count + failed_count} metric(s) could not be analyzed (likely due to insufficient data).\n"
            summary_msg += f"Results saved to {ANALYSIS_DIR}\n"
            summary_msg += "Click 'Refresh' to update status indicators, or click on '✓ Analyzed' labels to view results.\n"
            self.root.after(0, lambda msg=summary_msg: self._log_analysis(msg))
            
            # Refresh the metrics list to update status indicators
            self.root.after(0, self._refresh_analysis)
            
            if skipped_count > 0 or failed_count > 0:
                self.root.after(0, lambda: messagebox.showwarning(
                    "Analysis Complete with Warnings",
                    f"Successfully analyzed {successful_count} metric(s).\n"
                    f"{skipped_count} metric(s) were skipped and {failed_count} metric(s) failed "
                    f"(likely due to insufficient data).\n"
                    f"Check the log for details."
                ))
            else:
                self.root.after(0, lambda: messagebox.showinfo("Analysis Complete", f"Successfully analyzed {successful_count} metric(s)!"))
            
        except Exception as e:
            self.root.after(0, lambda e=e: self._log_analysis(f"Error during analysis: {e}\n"))
            self.root.after(0, lambda e=e: messagebox.showerror("Analysis Error", f"Error: {e}"))
        finally:
            self.is_analyzing = False
    
    def _generate_figures(self):
        """Generate selected figures."""
        if self.is_generating_figures:
            messagebox.showwarning("Generation Running", "Figure generation is already in progress!")
            return
        
        # Collect all figures to generate
        figures_to_generate = []
        
        # Gait figure
        if self.gait_figure_var.get():
            figures_to_generate.append(('gait', None))
        
        # Boxplot by feet attached
        if self.feet_boxplot_var.get():
            metric = self.feet_metric_var.get()
            if not metric:
                messagebox.showwarning("No Metric", "Please select a metric for the feet-attached boxplot.")
                return
            try:
                num_feet = int(self.num_feet_var.get())
                figures_to_generate.append(('feet_boxplot', (metric, num_feet)))
            except ValueError:
                messagebox.showwarning("Invalid Input", "Please enter a valid number of feet (0-6).")
                return
        
        # Boxplot Generator - either multi-parameter or individual
        selected_multi_metrics = [m for m, var in self.multi_metric_vars.items() if var.get()]
        if selected_multi_metrics:
            if self.multi_param_var.get():
                # Generate all in one figure
                figures_to_generate.append(('multi_param', selected_multi_metrics))
            else:
                # Generate separate boxplot for each selected metric
                for metric in selected_multi_metrics:
                    figures_to_generate.append(('single_boxplot', metric))
        
        if not figures_to_generate:
            messagebox.showwarning("No Selection", "Please select at least one figure to generate.")
            return
        
        self.is_generating_figures = True
        self.figures_thread = threading.Thread(target=self._generate_figures_thread, args=(figures_to_generate,), daemon=True)
        self.figures_thread.start()
    
    def _generate_figures_thread(self, figures_to_generate):
        """Generate figures in background thread."""
        try:
            total = len(figures_to_generate)
            self.root.after(0, lambda: self.figures_progress.config(maximum=total, value=0))
            self.root.after(0, lambda: self._log_figures(f"Generating {total} figure(s)...\n"))
            
            generated_count = 0
            failed_count = 0
            
            for i, (fig_type, fig_data) in enumerate(figures_to_generate, 1):
                display_name = ""
                expected_path = None
                
                try:
                    if fig_type == 'gait':
                        display_name = "Gait Pattern Heatmap"
                        expected_path = FIGURES_DIR / "Gait_Pattern_Heatmap.png"
                        self.root.after(0, lambda name=display_name, idx=i: self._log_figures(f"[{idx}/{total}] Creating: {name}"))
                        from master_figures import create_gait_figure
                        create_gait_figure()
                    
                    elif fig_type == 'feet_boxplot':
                        metric, num_feet = fig_data
                        metric_prefix, parameter_name = metric.split(":")
                        display_name = f"{parameter_name} ({num_feet} feet attached)"
                        expected_path = FIGURES_DIR / f"{parameter_name}_{num_feet}_feet_boxplot.png"
                        self.root.after(0, lambda name=display_name, idx=i: self._log_figures(f"[{idx}/{total}] Creating: {name}"))
                        from master_figures import create_boxplot_by_feet_attached
                        create_boxplot_by_feet_attached(metric, num_feet)
                    
                    elif fig_type == 'multi_param':
                        metrics = fig_data
                        param_names = '_'.join([m.split(':')[1] for m in metrics[:3]])
                        if len(metrics) > 3:
                            param_names += f'_and_{len(metrics)-3}_more'
                        display_name = f"Multi-Parameter Boxplot ({len(metrics)} parameters)"
                        expected_path = FIGURES_DIR / f"Multi_Parameter_Boxplot_{param_names}.png"
                        self.root.after(0, lambda name=display_name, idx=i: self._log_figures(f"[{idx}/{total}] Creating: {name}"))
                        from master_figures import create_multi_parameter_boxplot
                        create_multi_parameter_boxplot(metrics)
                    
                    elif fig_type == 'single_boxplot':
                        metric = fig_data
                        metric_prefix, parameter_name = metric.split(":")
                        display_name = f"Boxplot: {parameter_name}"
                        expected_path = FIGURES_DIR / f"{parameter_name}_boxplot.png"
                        self.root.after(0, lambda name=display_name, idx=i: self._log_figures(f"[{idx}/{total}] Creating: {name}"))
                        from master_figures import create_single_parameter_boxplot
                        create_single_parameter_boxplot(metric)
                    
                    # Check if file actually exists
                    if expected_path and expected_path.exists():
                        generated_count += 1
                        self.last_generated_figure = expected_path  # Track most recent
                        self.root.after(0, lambda name=display_name: self._log_figures(f"  ✓ Saved: {name}\n"))
                    else:
                        failed_count += 1
                        warning_msg = f"  ⚠ Warning: {display_name} - No figure file was created. This may be due to insufficient data.\n"
                        self.root.after(0, lambda msg=warning_msg: self._log_figures(msg))
                        
                except Exception as e:
                    failed_count += 1
                    self.root.after(0, lambda e=e, name=display_name: self._log_figures(f"  ✗ Error: {e}\n"))
                    import traceback
                    traceback.print_exc()
                
                self.root.after(0, lambda idx=i: self.figures_progress.config(value=idx))
            
            # Final summary
            summary_msg = f"\nFigure generation complete! Generated: {generated_count}, Failed/Warnings: {failed_count}\n"
            if failed_count > 0:
                summary_msg += f"⚠ Warning: {failed_count} figure(s) could not be generated (likely due to insufficient data).\n"
            summary_msg += f"Figures saved to {FIGURES_DIR}\n"
            self.root.after(0, lambda msg=summary_msg: self._log_figures(msg))
            
            # Update preview to show most recently generated figure
            if self.last_generated_figure and self.last_generated_figure.exists():
                self.root.after(0, self._preview_most_recent)
            
            self.root.after(0, lambda: self._update_preview_dropdown())
            
            if failed_count > 0:
                self.root.after(0, lambda: messagebox.showwarning(
                    "Generation Complete with Warnings", 
                    f"Generated {generated_count} figure(s) successfully.\n"
                    f"{failed_count} figure(s) could not be generated (likely due to insufficient data).\n"
                    f"Check the log for details."
                ))
            else:
                self.root.after(0, lambda: messagebox.showinfo("Generation Complete", f"Generated {generated_count} figure(s)!"))
            
        except Exception as e:
            self.root.after(0, lambda e=e: self._log_figures(f"Error during generation: {e}\n"))
            self.root.after(0, lambda e=e: messagebox.showerror("Generation Error", f"Error: {e}"))
        finally:
            self.is_generating_figures = False
    
    def _preview_figure(self, event=None):
        """Preview selected figure."""
        selected_name = self.preview_var.get()
        if not selected_name:
            return
        
        # Use the stored mapping if available (from most recent 10 figures)
        if hasattr(self, 'figure_preview_map') and selected_name in self.figure_preview_map:
            fig_path = self.figure_preview_map[selected_name]
        else:
            # Fallback: search for the file (for backward compatibility)
            fig_path = None
            
            # Check if it's gait figure
            if selected_name == "Gait Pattern Heatmap":
                fig_path = FIGURES_DIR / "Gait_Pattern_Heatmap.png"
            
            # Check if it's a multi-parameter boxplot
            elif "Multi-Parameter Boxplot" in selected_name:
                # Find the most recent multi-parameter boxplot
                multi_figs = list(FIGURES_DIR.glob("Multi_Parameter_Boxplot_*.png"))
                if multi_figs:
                    multi_figs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    fig_path = multi_figs[0]
            
            # Check if it's an individual boxplot
            elif selected_name.startswith("Boxplot: "):
                param_name = selected_name.replace("Boxplot: ", "").replace(" ", "_")
                fig_path = FIGURES_DIR / f"{param_name}_boxplot.png"
            
            # Check if it's a feet-attached boxplot
            else:
                # Try to find by name pattern
                search_name = selected_name.replace(" ", "_")
                potential_path = FIGURES_DIR / f"{search_name}.png"
                if potential_path.exists():
                    fig_path = potential_path
        
        if not fig_path or not fig_path.exists():
            messagebox.showwarning("Figure Not Found", f"Figure file not found for: {selected_name}")
            return
        
        # Clear previous preview
        for widget in self.preview_canvas_frame.winfo_children():
            widget.destroy()
        
        # Load and display image
        try:
            img = Image.open(fig_path)
            # Resize to fit preview area
            max_width = 600
            max_height = 500
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            label = ttk.Label(self.preview_canvas_frame, image=photo)
            label.image = photo  # Keep a reference
            label.pack(padx=5, pady=5)
        except Exception as e:
            messagebox.showerror("Preview Error", f"Could not load figure: {e}")
    
    def _open_analysis_dir(self):
        """Open analysis results directory."""
        if ANALYSIS_DIR.exists():
            if sys.platform == "win32":
                os.startfile(ANALYSIS_DIR)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(ANALYSIS_DIR)])
            else:
                subprocess.Popen(["xdg-open", str(ANALYSIS_DIR)])
        else:
            messagebox.showwarning("Directory Not Found", f"Directory does not exist: {ANALYSIS_DIR}")
    
    def _open_figures_dir(self):
        """Open figures directory."""
        if FIGURES_DIR.exists():
            if sys.platform == "win32":
                os.startfile(FIGURES_DIR)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(FIGURES_DIR)])
            else:
                subprocess.Popen(["xdg-open", str(FIGURES_DIR)])
        else:
            messagebox.showwarning("Directory Not Found", f"Directory does not exist: {FIGURES_DIR}")
    
    def _view_metric_results(self, metric):
        """View analysis results for a specific metric."""
        metric_prefix, parameter_name = metric.split(":")
        from master_analysis import map_metric_to_sheet
        sheet_name = map_metric_to_sheet(metric_prefix)
        results_dir = ANALYSIS_DIR / metric_prefix / parameter_name
        results_file = results_dir / "analysis_results.txt"
        boxplot_file = results_dir / f"{parameter_name}_boxplot.png"
        
        if not results_file.exists():
            messagebox.showinfo("No Results", f"Analysis results not found for:\n{metric}\n\nPlease run the analysis first.")
            return
        
        # Create a dialog to view results
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Analysis Results: {metric}")
        dialog.geometry("1000x700")
        
        # Top frame with metric name
        top_frame = ttk.Frame(dialog)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(top_frame, text=f"Metric: {metric}", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        
        # Main content frame - split into text and plot
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left side: Results text
        text_frame = ttk.LabelFrame(main_frame, text="Statistical Results")
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        results_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, font=("Courier", 10))
        results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Load results text
        with open(results_file, 'r', encoding='utf-8') as f:
            content = f.read()
        results_text.insert(1.0, content)
        results_text.config(state=tk.DISABLED)  # Make read-only
        
        # Right side: Boxplot
        plot_frame = ttk.LabelFrame(main_frame, text="Boxplot")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        if boxplot_file.exists():
            try:
                img = Image.open(boxplot_file)
                # Resize to fit
                max_width = 450
                max_height = 600
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(img)
                img_label = ttk.Label(plot_frame, image=photo)
                img_label.image = photo  # Keep a reference
                img_label.pack(padx=5, pady=5)
            except Exception as e:
                ttk.Label(plot_frame, text=f"Could not load boxplot:\n{e}", foreground="red").pack(padx=5, pady=5)
        else:
            ttk.Label(plot_frame, text="Boxplot not found", foreground="gray").pack(padx=5, pady=5)
        
        # Bottom buttons
        bottom_frame = ttk.Frame(dialog)
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)
        
        def open_results_dir():
            if results_dir.exists():
                if sys.platform == "win32":
                    os.startfile(results_dir)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(results_dir)])
                else:
                    subprocess.Popen(["xdg-open", str(results_dir)])
        
        ttk.Button(bottom_frame, text="Open Results Directory", command=open_results_dir).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Close", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _view_analysis_results(self):
        """Open dialog to select and view analysis results (legacy method)."""
        # Show message directing to click on analyzed metrics
        messagebox.showinfo(
            "View Results",
            "To view analysis results:\n\n"
            "1. Look for metrics marked '✓ Analyzed' in green\n"
            "2. Click on the '✓ Analyzed' label or double-click the metric\n"
            "3. Or use the dropdown below to select a metric"
        )
    
    def _log_analysis(self, message):
        """Log message to analysis log area."""
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.analysis_log.insert(tk.END, f"{timestamp} {message}")
        self.analysis_log.see(tk.END)
    
    def _log_figures(self, message):
        """Log message to figures log area."""
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.figures_log.insert(tk.END, f"{timestamp} {message}")
        self.figures_log.see(tk.END)


def main():
    root = tk.Tk()
    app = AnalysisFiguresGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
