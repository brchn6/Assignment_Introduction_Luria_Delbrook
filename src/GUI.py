"""
Graphical User Interface for Luria-Delbrück Experiment Simulator
This script provides a simple GUI for running the Luria-Delbrück experiment
path: ./Assignment_Introduction_Luria_Delbrook/src/GUI.py
"""

"""
Graphical User Interface for Luria-Delbrück Experiment Simulator
This script provides a simple GUI for running the Luria-Delbrück experiment
path: ./Assignment_Introduction_Luria_Delbrook/src/GUI.py
"""

"""
Graphical User Interface for Luria-Delbrück Experiment Simulator
This script provides a simple GUI for running the Luria-Delbrück experiment
path: ./Assignment_Introduction_Luria_Delbrook/src/GUI.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import os
import sys
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import subprocess
import queue
import pandas as pd
from datetime import datetime

class LuriaDelbrueckGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Luria-Delbrück Experiment Simulator")
        self.root.geometry("1000x700")
        self.root.minsize(900, 650)
        
        # Find the main script path
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.main_script = os.path.join(self.script_dir, "main.py")
        
        # Check if main.py exists
        if not os.path.exists(self.main_script):
            self.show_error_and_browse_for_script()
        
        # Create a queue for thread-safe communication
        self.queue = queue.Queue()
        
        # Set up the main frames
        self.create_frames()
        self.create_parameter_widgets()
        self.create_results_area()
        self.create_visualization_area()
        
        # Set default values
        self.set_defaults()
        
        # Periodically check the queue
        self.root.after(100, self.check_queue)

    def show_error_and_browse_for_script(self):
        """Show error message and let user browse for main.py"""
        message_frame = ttk.Frame(self.root, padding=20)
        message_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(message_frame, text="Error: main.py not found!", font=('Arial', 14, 'bold')).pack(pady=10)
        ttk.Label(message_frame, text=f"Could not find main.py in {self.script_dir}").pack(pady=5)
        ttk.Label(message_frame, text="Please locate the main.py script:").pack(pady=20)
        
        def browse_for_script():
            file_path = filedialog.askopenfilename(
                title="Select main.py",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")]
            )
            if file_path and os.path.exists(file_path):
                self.main_script = file_path
                self.script_dir = os.path.dirname(file_path)
                message_frame.destroy()
                # Restart initialization
                self.create_frames()
                self.create_parameter_widgets()
                self.create_results_area()
                self.create_visualization_area()
                self.set_defaults()
                self.root.after(100, self.check_queue)
        
        ttk.Button(message_frame, text="Browse for main.py", command=browse_for_script).pack(pady=10)

    def create_frames(self):
        # Main frame with paned window for resizable sections
        self.main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame for parameters
        self.params_frame = ttk.LabelFrame(self.main_frame, text="Simulation Parameters")
        self.main_frame.add(self.params_frame, weight=30)
        
        # Right frame for results
        self.right_pane = ttk.PanedWindow(self.main_frame, orient=tk.VERTICAL)
        self.main_frame.add(self.right_pane, weight=70)
        
        # Split right pane into results and visualization
        self.results_frame = ttk.LabelFrame(self.right_pane, text="Results")
        self.visualization_frame = ttk.LabelFrame(self.right_pane, text="Visualization")
        self.right_pane.add(self.results_frame, weight=40)
        self.right_pane.add(self.visualization_frame, weight=60)

    def create_parameter_widgets(self):
        # Script path display and edit
        script_frame = ttk.Frame(self.params_frame)
        script_frame.grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=5, padx=5)
        
        ttk.Label(script_frame, text="Script:").pack(side=tk.LEFT)
        self.script_path_var = tk.StringVar(value=self.main_script)
        ttk.Entry(script_frame, textvariable=self.script_path_var, width=25).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        ttk.Button(script_frame, text="Browse", command=self.browse_script).pack(side=tk.LEFT)
        
        # Model selection
        ttk.Label(self.params_frame, text="Mutation Model:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        self.model_var = tk.StringVar()
        model_combo = ttk.Combobox(self.params_frame, textvariable=self.model_var)
        model_combo['values'] = ('random', 'induced', 'combined', 'all')
        model_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Number of cultures
        ttk.Label(self.params_frame, text="Number of Cultures:").grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        self.cultures_var = tk.StringVar()
        ttk.Entry(self.params_frame, textvariable=self.cultures_var).grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Initial population size
        ttk.Label(self.params_frame, text="Initial Population Size:").grid(row=3, column=0, sticky=tk.W, pady=5, padx=5)
        self.pop_size_var = tk.StringVar()
        ttk.Entry(self.params_frame, textvariable=self.pop_size_var).grid(row=3, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Number of generations
        ttk.Label(self.params_frame, text="Number of Generations:").grid(row=4, column=0, sticky=tk.W, pady=5, padx=5)
        self.generations_var = tk.StringVar()
        ttk.Entry(self.params_frame, textvariable=self.generations_var).grid(row=4, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Mutation rate
        ttk.Label(self.params_frame, text="Mutation Rate:").grid(row=5, column=0, sticky=tk.W, pady=5, padx=5)
        self.mut_rate_var = tk.StringVar()
        ttk.Entry(self.params_frame, textvariable=self.mut_rate_var).grid(row=5, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Induced mutation rate (for combined model)
        ttk.Label(self.params_frame, text="Induced Mutation Rate:").grid(row=6, column=0, sticky=tk.W, pady=5, padx=5)
        self.induced_rate_var = tk.StringVar()
        self.induced_rate_entry = ttk.Entry(self.params_frame, textvariable=self.induced_rate_var)
        self.induced_rate_entry.grid(row=6, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Number of genes
        ttk.Label(self.params_frame, text="Number of Genes:").grid(row=7, column=0, sticky=tk.W, pady=5, padx=5)
        self.genes_var = tk.StringVar()
        ttk.Entry(self.params_frame, textvariable=self.genes_var).grid(row=7, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Output directory
        ttk.Label(self.params_frame, text="Results Directory:").grid(row=8, column=0, sticky=tk.W, pady=5, padx=5)
        self.results_dir_var = tk.StringVar()
        results_frame = ttk.Frame(self.params_frame)
        results_frame.grid(row=8, column=1, sticky=tk.W, pady=5, padx=5)
        ttk.Entry(results_frame, textvariable=self.results_dir_var, width=15).pack(side=tk.LEFT)
        ttk.Button(results_frame, text="Browse", command=self.browse_output_dir).pack(side=tk.LEFT, padx=5)
        
        # Use parallel processing
        self.parallel_var = tk.BooleanVar()
        ttk.Checkbutton(self.params_frame, text="Use Parallel Processing", variable=self.parallel_var).grid(
            row=9, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)
        
        # Logging level
        ttk.Label(self.params_frame, text="Log Level:").grid(row=10, column=0, sticky=tk.W, pady=5, padx=5)
        self.log_level_var = tk.StringVar()
        log_combo = ttk.Combobox(self.params_frame, textvariable=self.log_level_var)
        log_combo['values'] = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_combo.grid(row=10, column=1, sticky=tk.W, pady=5, padx=5)
        
        # Optimal LD parameters
        self.optimal_ld_var = tk.BooleanVar()
        ttk.Checkbutton(self.params_frame, text="Use Optimal L-D Parameters", 
                        variable=self.optimal_ld_var,
                        command=self.toggle_optimal_params).grid(
            row=11, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)
        
        # Run button
        self.run_button = ttk.Button(self.params_frame, text="Run Simulation", command=self.run_simulation)
        self.run_button.grid(row=12, column=0, columnspan=2, pady=10)
        
        # Progress indicator
        self.progress = ttk.Progressbar(self.params_frame, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
        self.progress.grid(row=13, column=0, columnspan=2, pady=10, padx=5, sticky=tk.EW)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(self.params_frame, textvariable=self.status_var).grid(row=14, column=0, columnspan=2, pady=5)
        
        # Advanced options expander
        self.advanced_frame = ttk.LabelFrame(self.params_frame, text="Advanced Options")
        self.advanced_frame.grid(row=15, column=0, columnspan=2, sticky=tk.EW, pady=10, padx=5)
        
        # Profile checkbox
        self.profile_var = tk.BooleanVar()
        ttk.Checkbutton(self.advanced_frame, text="Run with Profiling", 
                       variable=self.profile_var).pack(anchor=tk.W, pady=5, padx=5)
        
        # Optimize NumPy checkbox
        self.optimize_numpy_var = tk.BooleanVar()
        ttk.Checkbutton(self.advanced_frame, text="Optimize NumPy",
                       variable=self.optimize_numpy_var).pack(anchor=tk.W, pady=5, padx=5)
        
        # Number of processes
        process_frame = ttk.Frame(self.advanced_frame)
        process_frame.pack(anchor=tk.W, pady=5, padx=5, fill=tk.X)
        ttk.Label(process_frame, text="Number of Processes:").pack(side=tk.LEFT)
        self.processes_var = tk.StringVar()
        ttk.Entry(process_frame, textvariable=self.processes_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(process_frame, text="(leave empty for auto)").pack(side=tk.LEFT)

    def browse_script(self):
        """Open dialog to select main.py script"""
        file_path = filedialog.askopenfilename(
            initialdir=self.script_dir,
            title="Select main.py script",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if file_path and os.path.exists(file_path):
            self.main_script = file_path
            self.script_path_var.set(file_path)
            self.script_dir = os.path.dirname(file_path)

    def create_results_area(self):
        # Create text widget for displaying results
        self.results_text = scrolledtext.ScrolledText(self.results_frame, wrap=tk.WORD, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Buttons for saving and loading results
        button_frame = ttk.Frame(self.results_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Clear", command=self.clear_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Results Directory", command=self.load_results_dir).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="View Report", command=self.load_detailed_report).pack(side=tk.LEFT, padx=5)

    def create_visualization_area(self):
        # Create matplotlib figure for displaying visualizations
        self.fig = plt.Figure(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.visualization_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Visualization control buttons
        control_frame = ttk.Frame(self.visualization_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Visualization type selection
        ttk.Label(control_frame, text="Visualization:").pack(side=tk.LEFT, padx=5)
        self.vis_type_var = tk.StringVar()
        vis_combo = ttk.Combobox(control_frame, textvariable=self.vis_type_var, width=25)
        vis_combo['values'] = (
            'Standard Distribution', 
            'Log-Scale Distribution',
            'CCDF Plot',
            'Model Comparison',
            'Theoretical Comparison',
            'Variance-to-Mean Ratio',
            'VMR Comparison'
        )
        vis_combo.pack(side=tk.LEFT, padx=5)
        vis_combo.bind('<<ComboboxSelected>>', self.update_visualization)
        
        # Model selection for visualization (when results from multiple models are available)
        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.vis_model_var = tk.StringVar()
        self.vis_model_combo = ttk.Combobox(control_frame, textvariable=self.vis_model_var, width=10)
        self.vis_model_combo['values'] = ('random', 'induced', 'combined')
        self.vis_model_combo.pack(side=tk.LEFT, padx=5)
        self.vis_model_combo.bind('<<ComboboxSelected>>', self.update_visualization)
        
        # Refresh visualization button
        ttk.Button(control_frame, text="Refresh", command=self.refresh_visualization).pack(side=tk.LEFT, padx=10)
        
        # Save visualization button
        ttk.Button(control_frame, text="Save Image", command=self.save_visualization).pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="Open Results Folder", command=self.open_results_folder).pack(side=tk.RIGHT, padx=5)

    def refresh_visualization(self):
        """Force refresh of the current visualization"""
        if not self.simulation_results:
            self.status_var.set("No data to visualize")
            return
            
        # Update visualization based on current selections
        self.update_visualization(None)
        self.status_var.set("Visualization refreshed")
        
        # Make sure the figure is properly displayed
        self.fig.tight_layout()
        self.canvas.draw_idle()
        
        # If there are issues with rendering, try stronger refresh methods
        try:
            self.canvas.flush_events()
            self.visualization_frame.update()
        except:
            pass
        
    def set_defaults(self):
        """Set default values for all parameters"""
        self.model_var.set("random")
        self.cultures_var.set("1000")
        self.pop_size_var.set("10")
        self.generations_var.set("5")
        self.mut_rate_var.set("0.1")
        self.induced_rate_var.set("0.1")
        self.genes_var.set("1")
        
        # Set default results directory to a 'results' subdirectory in the script directory
        default_results_dir = os.path.join(self.script_dir, "..", "Results")
        self.results_dir_var.set(default_results_dir)
        
        self.parallel_var.set(True)
        self.log_level_var.set("INFO")
        self.optimal_ld_var.set(False)
        self.profile_var.set(False)
        self.optimize_numpy_var.set(False)
        self.processes_var.set("")
        
        # Initialize visualization defaults
        self.vis_type_var.set("Standard Distribution")
        self.vis_model_var.set("random")
        
        # Update UI based on defaults
        self.on_model_change(None)
        self.toggle_optimal_params()
        
        # Initialize results storage
        self.current_results_dir = None
        self.simulation_results = {}
        self.statistics = {}

    def on_model_change(self, event):
        """Enable/disable fields based on selected model"""
        model = self.model_var.get()
        
        if model == "combined" or model == "all":
            self.induced_rate_entry.config(state="normal")
        else:
            self.induced_rate_entry.config(state="disabled")

    def toggle_optimal_params(self):
        """Set optimal parameters for Luria-Delbrück demonstration"""
        if self.optimal_ld_var.get():
            # Store current values
            self.stored_values = {
                'cultures': self.cultures_var.get(),
                'pop_size': self.pop_size_var.get(),
                'generations': self.generations_var.get(),
                'mut_rate': self.mut_rate_var.get(),
                'induced_rate': self.induced_rate_var.get()
            }
            
            # Set optimal values
            self.cultures_var.set("5000")
            self.pop_size_var.set("100")
            self.generations_var.set("20")
            self.mut_rate_var.set("0.0001")
            self.induced_rate_var.set("0.05")
            
            # Update the status to inform the user
            self.status_var.set("Using optimal parameters for Luria-Delbrück demonstration")
        else:
            # Restore previous values if they exist
            if hasattr(self, 'stored_values'):
                self.cultures_var.set(self.stored_values['cultures'])
                self.pop_size_var.set(self.stored_values['pop_size'])
                self.generations_var.set(self.stored_values['generations'])
                self.mut_rate_var.set(self.stored_values['mut_rate'])
                self.induced_rate_var.set(self.stored_values['induced_rate'])
                
                # Update status
                self.status_var.set("Using custom parameters")

    def browse_output_dir(self):
        """Open dialog to select output directory"""
        directory = filedialog.askdirectory(initialdir=self.script_dir)
        if directory:
            self.results_dir_var.set(directory)

    def run_simulation(self):
        """Run the simulation with the specified parameters"""
        # Verify script path exists
        script_path = self.script_path_var.get()
        if not os.path.exists(script_path):
            self.results_text.insert(tk.END, f"Error: Script not found at {script_path}\n", "error")
            self.results_text.tag_configure("error", foreground="red")
            self.status_var.set("Error: Script not found")
            return
                
        # Validate numeric inputs
        try:
            int(self.cultures_var.get())
            int(self.pop_size_var.get())
            int(self.generations_var.get())
            int(self.genes_var.get())
            float(self.mut_rate_var.get())
            float(self.induced_rate_var.get())
            if self.processes_var.get() and not self.processes_var.get().isdigit():
                raise ValueError("Number of processes must be a positive integer")
        except ValueError as e:
            self.results_text.insert(tk.END, f"Invalid parameter value: {str(e)}\n", "error")
            self.results_text.tag_configure("error", foreground="red")
            self.status_var.set("Error: Invalid parameter value")
            return
        
        # Clear previous simulation results before starting a new run
        self.simulation_results = {}
        self.statistics = {}
        
        # Clear any existing visualizations
        self.fig.clear()
        self.canvas.draw()
        
        # Disable the run button and show progress
        self.run_button.config(state="disabled")
        self.progress.start(10)
        self.status_var.set("Running simulation...")
        
        # Build command arguments
        cmd_args = self.build_command_args()
        
        # Clear results display
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Starting simulation with parameters:\n{' '.join(cmd_args)}\n\n")
        self.results_text.see(tk.END)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.results_text.insert(tk.END, f"Simulation started at: {timestamp}\n\n")
        
        # Run in a separate thread to avoid freezing the UI
        thread = threading.Thread(target=self.run_simulation_thread, args=(cmd_args,))
        thread.daemon = True
        thread.start()

    def build_command_args(self):
        """Build command line arguments from GUI inputs"""
        script_path = self.script_path_var.get()
        model = self.model_var.get()
        cultures = self.cultures_var.get()
        pop_size = self.pop_size_var.get()
        generations = self.generations_var.get()
        mut_rate = self.mut_rate_var.get()
        induced_rate = self.induced_rate_var.get()
        genes = self.genes_var.get()
        results_dir = self.results_dir_var.get()
        log_level = self.log_level_var.get()
        use_parallel = self.parallel_var.get()
        processes = self.processes_var.get()
        profile = self.profile_var.get()
        optimize_numpy = self.optimize_numpy_var.get()
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            try:
                os.makedirs(results_dir, exist_ok=True)
                self.results_text.insert(tk.END, f"Created results directory: {results_dir}\n")
            except Exception as e:
                self.results_text.insert(tk.END, f"Error creating results directory: {str(e)}\n", "error")
        
        # Get the python executable to use
        python_exec = sys.executable
        
        # Build command arguments list
        cmd_args = [python_exec, script_path,
                    "--model", model,
                    "--num_cultures", cultures,
                    "--initial_population_size", pop_size,
                    "--num_generations", generations,
                    "--mutation_rate", mut_rate,
                    "--n_genes", genes,
                    "--Results_path", results_dir,
                    "--log_level", log_level]
        
        # Add optional arguments
        if model == "combined" or model == "all":
            cmd_args.extend(["--induced_mutation_rate", induced_rate])
        
        if use_parallel:
            cmd_args.append("--use_parallel")
            if processes:
                cmd_args.extend(["--n_processes", processes])
        
        if profile:
            cmd_args.append("--profile")
        
        if optimize_numpy:
            cmd_args.append("--optimize_numpy")
            
        if self.optimal_ld_var.get():
            cmd_args.append("--optimal_ld")
            
        return cmd_args

    def run_simulation_thread(self, cmd_args):
        """Execute simulation in a separate thread"""
        try:
            # Create the process and capture output
            process = subprocess.Popen(cmd_args, 
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    bufsize=1,
                                    universal_newlines=True)
            
            # Track if we're running the "all" model
            is_all_models = False
            for i, arg in enumerate(cmd_args):
                if arg == "--model" and i+1 < len(cmd_args) and cmd_args[i+1] == "all":
                    is_all_models = True
                    break
            
            # Read and display output in real-time
            for line in iter(process.stdout.readline, ''):
                self.queue.put(('output', line))
                
                # Parse statistics from output for later visualization
                self.parse_output_statistics(line)
            
            # Get any error output
            for line in iter(process.stderr.readline, ''):
                self.queue.put(('error', line))
            
            # Wait for process to complete
            process.wait()
            
            # Store output directory for visualization
            if process.returncode == 0:
                # Try to find the output directory from the logs
                for arg_idx, arg in enumerate(cmd_args):
                    if arg == "--Results_path" and arg_idx + 1 < len(cmd_args):
                        base_dir = cmd_args[arg_idx + 1]
                        # Find the most recent directory within results_dir
                        if os.path.exists(base_dir):
                            subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                                    if os.path.isdir(os.path.join(base_dir, d))]
                            if subdirs:
                                latest_dir = max(subdirs, key=os.path.getmtime)
                                self.current_results_dir = latest_dir
                                self.queue.put(('status', f"Results saved to: {latest_dir}"))
                                
                                # Load results for visualization
                                self.load_simulation_results(self.current_results_dir)
                                
                                # Force visualization refresh after loading results
                                self.queue.put(('refresh_vis', ""))
                                
                                # For 'all' model, automatically switch to comparison view
                                if is_all_models:
                                    self.queue.put(('set_comparison_view', ""))
                        break
                                
                self.queue.put(('status', "Simulation completed successfully"))
            else:
                self.queue.put(('status', f"Simulation failed with code {process.returncode}"))
            
            # Send message to enable UI elements
            self.queue.put(('complete', ""))
            
        except Exception as e:
            self.queue.put(('error', f"Error: {str(e)}"))
            self.queue.put(('status', "Simulation failed"))
            self.queue.put(('complete', ""))    
    
    def parse_output_statistics(self, line):
            """Parse statistics from the simulation output"""
            # Parse mean, variance, coefficient of variation
            if "Mean: " in line:
                parts = line.strip().split()
                model = None
                value = None
                
                # Look for model name
                if "random" in line.lower():
                    model = "random"
                elif "induced" in line.lower():
                    model = "induced"
                elif "combined" in line.lower():
                    model = "combined"
                    
                # Look for statistic type and value
                for i, part in enumerate(parts):
                    if part == "mean:" and i+1 < len(parts):
                        stat_type = "mean"
                        try:
                            value = float(parts[i+1])
                        except ValueError:
                            pass
                    elif part == "variance:" and i+1 < len(parts):
                        stat_type = "variance"
                        try:
                            value = float(parts[i+1])
                        except ValueError:
                            pass
                    elif "cv" in part.lower() and i+1 < len(parts):
                        stat_type = "cv"
                        try:
                            value = float(parts[i+1])
                        except ValueError:
                            pass
                    elif "vmr" in part.lower() and i+1 < len(parts):
                        stat_type = "vmr"
                        try:
                            value = float(parts[i+1])
                        except ValueError:
                            pass
                            
                # Store the statistic if we found model and value
                if model and value is not None:
                    if model not in self.statistics:
                        self.statistics[model] = {}
                    self.statistics[model][stat_type] = value

    def check_queue(self):
        """Check for messages from the simulation thread"""
        try:
            while True:
                msg_type, msg = self.queue.get_nowait()
                
                if msg_type == 'output':
                    self.results_text.insert(tk.END, msg)
                    self.results_text.see(tk.END)
                elif msg_type == 'error':
                    self.results_text.insert(tk.END, f"ERROR: {msg}\n", "error")
                    self.results_text.tag_configure("error", foreground="red")
                    self.results_text.see(tk.END)
                elif msg_type == 'status':
                    self.status_var.set(msg)
                elif msg_type == 'refresh_vis':
                    # Force visualization update
                    if self.simulation_results:
                        self.update_visualization(None)
                elif msg_type == 'set_comparison_view':
                    # Automatically set visualization to comparison view for 'all' model
                    if len(self.simulation_results) > 1:
                        self.vis_type_var.set("Model Comparison")
                        self.update_visualization(None)
                elif msg_type == 'complete':
                    # Re-enable UI elements
                    self.run_button.config(state="normal")
                    self.progress.stop()
                    
                    # Add timestamp for completion
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.results_text.insert(tk.END, f"\nSimulation completed at: {timestamp}\n")
                    
                    # Update visualization options based on results
                    self.update_visualization_options()
                
                # Mark as done
                self.queue.task_done()
                
        except queue.Empty:
            # No more messages, schedule the next check
            self.root.after(100, self.check_queue)    
            
    def load_simulation_results(self, directory):
            """Load simulation results from files for visualization"""
            try:
                # Clear previous results
                self.simulation_results = {}
                self.statistics = {}
                
                # Clear any existing visualization
                self.fig.clear()
                self.canvas.draw()
                
                # Look for CSV files with survivor data
                for model in ['random', 'induced', 'combined']:
                    csv_path = os.path.join(directory, f"survivors_{model}.csv")
                    if os.path.exists(csv_path):
                        data = np.loadtxt(csv_path, delimiter=',')
                        self.simulation_results[model] = data
                        
                        # Try to load statistics from model_comparison_report.txt
                        self.load_statistics_from_report(directory)
                
                # If we found any results, update visualization
                if self.simulation_results:
                    self.queue.put(('status', f"Loaded results for {', '.join(self.simulation_results.keys())}"))
                    self.update_visualization_options()
                    
                    # Set default visualization to first available model
                    if self.vis_model_var.get() not in self.simulation_results:
                        self.vis_model_var.set(next(iter(self.simulation_results.keys())))
                    
                    # Force update visualization with a small delay to ensure UI is ready
                    self.root.after(100, lambda: self.update_visualization(None))
            except Exception as e:
                self.queue.put(('error', f"Error loading results: {str(e)}"))

    def load_statistics_from_report(self, directory):
        """Extract statistics from the report files"""
        # First try the detailed Luria-Delbrück report
        ld_report_path = os.path.join(directory, "luria_delbrueck_detailed_report.txt")
        if os.path.exists(ld_report_path):
            try:
                with open(ld_report_path, 'r') as f:
                    content = f.read()
                
                # Parse statistics for each model
                for model in ['random', 'induced', 'combined']:
                    if model not in self.statistics:
                        self.statistics[model] = {}
                    
                    # Look for mean, variance, and VMR for each model
                    model_section = None
                    if model == 'random':
                        model_section = content.split("Random (Darwinian) Model:")[1].split("Induced (Lamarckian) Model:")[0]
                    elif model == 'induced':
                        split_content = content.split("Induced (Lamarckian) Model:")[1]
                        if "3. Comparative Analysis:" in split_content:
                            model_section = split_content.split("3. Comparative Analysis:")[0]
                        else:
                            model_section = split_content
                    elif model == 'combined' and "Combined Model" in content:
                        # Extract combined model section if it exists
                        try:
                            model_section = content.split("Combined Model")[1].split("\n\n")[0]
                        except:
                            pass
                    
                    if model_section:
                        # Extract mean
                        if "Mean mutants per culture:" in model_section:
                            try:
                                mean_line = [line for line in model_section.split('\n') if "Mean mutants per culture:" in line][0]
                                mean_val = float(mean_line.split(':')[1].strip())
                                self.statistics[model]['mean'] = mean_val
                            except:
                                pass
                        
                        # Extract variance
                        if "Variance:" in model_section:
                            try:
                                var_line = [line for line in model_section.split('\n') if "Variance:" in line][0]
                                var_val = float(var_line.split(':')[1].strip())
                                self.statistics[model]['variance'] = var_val
                            except:
                                pass
                        
                        # Extract VMR
                        if "Variance-to-Mean Ratio:" in model_section:
                            try:
                                vmr_line = [line for line in model_section.split('\n') if "Variance-to-Mean Ratio:" in line][0]
                                vmr_val = float(vmr_line.split(':')[1].strip())
                                self.statistics[model]['vmr'] = vmr_val
                            except:
                                pass
                    
            except Exception as e:
                print(f"Error parsing LD report: {str(e)}")
        
        # Also try the model comparison report
        report_path = os.path.join(directory, "model_comparison_report.txt")
        if os.path.exists(report_path) and not self.statistics:
            try:
                with open(report_path, 'r') as f:
                    lines = f.readlines()
                
                # Find the statistics table
                table_start = None
                for i, line in enumerate(lines):
                    if "Statistic" in line and "Random" in line and "Induced" in line:
                        table_start = i
                        break
                
                if table_start:
                    # Parse each stat
                    stats_map = {
                        'Mean': 'mean',
                        'Variance': 'variance',
                        'Coef. of Variation': 'cv'
                    }
                    
                    for i in range(table_start + 2, len(lines)):
                        line = lines[i].strip()
                        if not line or "=" in line:
                            break
                            
                        parts = line.split()
                        if len(parts) >= 4:
                            stat_name = parts[0]
                            if stat_name in stats_map:
                                key = stats_map[stat_name]
                                # Values are in columns for random, induced, combined
                                try:
                                    random_val = float(parts[1])
                                    induced_val = float(parts[2])
                                    combined_val = float(parts[3]) if len(parts) > 3 else None
                                    
                                    if 'random' not in self.statistics:
                                        self.statistics['random'] = {}
                                    if 'induced' not in self.statistics:
                                        self.statistics['induced'] = {}
                                    if 'combined' not in self.statistics:
                                        self.statistics['combined'] = {}
                                        
                                    self.statistics['random'][key] = random_val
                                    self.statistics['induced'][key] = induced_val
                                    if combined_val is not None:
                                        self.statistics['combined'][key] = combined_val
                                except:
                                    pass
            except Exception as e:
                print(f"Error parsing comparison report: {str(e)}")
                
        # If we have mean and variance but no VMR, calculate it
        for model in self.statistics:
            if 'mean' in self.statistics[model] and 'variance' in self.statistics[model] and 'vmr' not in self.statistics[model]:
                mean = self.statistics[model]['mean']
                if mean > 0:
                    self.statistics[model]['vmr'] = self.statistics[model]['variance'] / mean

    def update_visualization_options(self):
        """Update the visualization options based on available results"""
        if self.simulation_results:
            # Update model selection combobox
            available_models = list(self.simulation_results.keys())
            self.vis_model_combo['values'] = available_models
            
            # If multiple models are available, enable comparison options
            if len(available_models) > 1:
                all_values = list(self.vis_type_var['values'])
                comparison_options = ['Model Comparison', 'VMR Comparison']
                
                # Add comparison options if not already present
                new_values = list(all_values)
                for option in comparison_options:
                    if option not in all_values:
                        new_values.append(option)
                
                self.vis_type_var['values'] = new_values
                
                # If 'all' model was run, automatically switch to comparison view
                if len(available_models) >= 2 and 'random' in available_models and 'induced' in available_models:
                    self.status_var.set("Multiple models available - Showing comparison view")

    def update_visualization(self, event):
        """Update the visualization based on selected type and model"""
        if not self.simulation_results:
            return
            
        vis_type = self.vis_type_var.get()
        model = self.vis_model_var.get()
        
        # Clear the figure
        self.fig.clear()
        
        # Get data for the selected model
        if vis_type != 'Model Comparison' and vis_type != 'VMR Comparison' and model in self.simulation_results:
            data = self.simulation_results[model]
            
            # Create appropriate visualization
            if vis_type == 'Standard Distribution':
                self.plot_standard_distribution(data, model)
            elif vis_type == 'Log-Scale Distribution':
                self.plot_log_scale_distribution(data, model)
            elif vis_type == 'CCDF Plot':
                self.plot_ccdf(data, model)
            elif vis_type == 'Theoretical Comparison':
                self.plot_theoretical_comparison(data, model)
            elif vis_type == 'Variance-to-Mean Ratio':
                self.plot_vmr(model)
        elif vis_type == 'Model Comparison' and len(self.simulation_results) > 1:
            self.plot_model_comparison()
        elif vis_type == 'VMR Comparison' and len(self.simulation_results) > 1:
            self.plot_vmr_comparison()
        
        # Refresh the canvas
        self.canvas.draw()

    def plot_standard_distribution(self, data, model):
        """Plot standard histogram of survivor counts"""
        ax = self.fig.add_subplot(111)
        ax.hist(data, bins='auto', alpha=0.7, color='royalblue')
        ax.set_xlabel("Number of Resistant Survivors")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{model.capitalize()} Model: Distribution of Resistant Survivors")
        
        # Add statistics if available
        if model in self.statistics and 'mean' in self.statistics[model]:
            stats_text = f"Mean: {self.statistics[model]['mean']:.2f}\n"
            if 'variance' in self.statistics[model]:
                stats_text += f"Variance: {self.statistics[model]['variance']:.2f}\n"
            if 'vmr' in self.statistics[model]:
                stats_text += f"VMR: {self.statistics[model]['vmr']:.2f}"
                
            # Add text box with statistics
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        self.fig.tight_layout()

    def plot_log_scale_distribution(self, data, model):
        """Plot log-scale histogram of survivor counts"""
        ax = self.fig.add_subplot(111)
        
        if max(data) > 0:
            # Add a small value to handle zeros when using log scale
            nonzero_data = [x + 0.1 for x in data if x > 0]
            if nonzero_data:
                bins = np.logspace(np.log10(min(nonzero_data)), np.log10(max(nonzero_data)), 20)
                ax.hist(nonzero_data, bins=bins, alpha=0.7, color='forestgreen')
                ax.set_xscale('log')
                ax.set_xlabel("Number of Resistant Survivors (log scale)")
                ax.set_ylabel("Frequency")
                ax.set_title(f"{model.capitalize()} Model: Log-Scale Distribution")
                
                # Add explanation text
                if model == 'random':
                    explanation = "Wide spread on log scale is characteristic\nof Luria-Delbrück distribution"
                    ax.text(0.05, 0.95, explanation, transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            else:
                ax.text(0.5, 0.5, "No non-zero data available for log scale", 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No data available for log scale", 
                   ha='center', va='center', transform=ax.transAxes)
        
        self.fig.tight_layout()

    def plot_ccdf(self, data, model):
        """Plot complementary cumulative distribution function"""
        ax = self.fig.add_subplot(111)
        
        sorted_data = np.sort(data)
        ccdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        ax.step(sorted_data, ccdf, where='post', color='darkorange', linewidth=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Number of Resistant Survivors (log scale)")
        ax.set_ylabel("P(X > x) (log scale)")
        ax.set_title(f"{model.capitalize()} Model: Complementary Cumulative Distribution Function")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        # Add explanation text
        if model == 'random':
            explanation = "A straight line on this log-log plot indicates\na power-law-like distribution, characteristic\nof the Luria-Delbrück distribution"
            ax.text(0.05, 0.05, explanation, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        elif model == 'induced':
            explanation = "A curved line on this log-log plot is\ncharacteristic of Poisson-like distributions\nexpected for induced mutations"
            ax.text(0.05, 0.05, explanation, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        self.fig.tight_layout()

    def plot_theoretical_comparison(self, data, model):
        """Plot comparison with theoretical distribution"""
        ax = self.fig.add_subplot(111)
        
        # Generate theoretical data for comparison
        if model == "random":
            # Simulate Luria-Delbrück distribution (simplified)
            # Using log-normal as an approximation
            mean_log, sigma_log = 0, 1  # Parameters can be adjusted
            theoretical_data = np.random.lognormal(mean_log, sigma_log, size=10000)
            # Scale to match mean of empirical data
            mean_data = np.mean(data)
            mean_theo = np.mean(theoretical_data)
            scaling_factor = mean_data / mean_theo if mean_theo > 0 else 1
            theoretical_data = theoretical_data * scaling_factor
            title = "Random Model vs. Theoretical Luria-Delbrück Distribution"
            theory_label = "Theoretical Luria-Delbrück (log-normal approx.)"
        else:
            # Simulate Poisson distribution for induced model
            mean_data = np.mean(data)
            theoretical_data = np.random.poisson(mean_data, size=10000)
            title = f"{model.capitalize()} Model vs. Theoretical Poisson Distribution"
            theory_label = "Theoretical Poisson distribution"
        
        # Use kernel density estimation for smooth curves
        try:
            from scipy.stats import gaussian_kde
            
            # For empirical data
            if len(data) > 1:
                kde_data = gaussian_kde(data)
                x = np.linspace(0, max(data) * 1.2, 1000)
                y_data = kde_data(x)
                ax.plot(x, y_data, color='blue', linewidth=2, label="Simulation data")
            else:
                # Not enough data for KDE
                ax.hist(data, bins=20, alpha=0.4, color='blue', density=True, label="Simulation data")
            
            # For theoretical data
            kde_theory = gaussian_kde(theoretical_data)
            x_theory = np.linspace(0, max(theoretical_data) * 1.2, 1000)
            y_theory = kde_theory(x_theory)
            ax.plot(x_theory, y_theory, color='red', linestyle='--', linewidth=2, label=theory_label)
            
        except Exception as e:
            # Fallback if KDE fails
            print(f"KDE failed: {str(e)}")
            ax.hist(data, bins=30, alpha=0.4, color='blue', density=True, label="Simulation data")
            ax.hist(theoretical_data, bins=30, alpha=0.4, color='red', density=True, label=theory_label)
        
        ax.set_xlabel("Number of Resistant Survivors")
        ax.set_ylabel("Probability Density")
        ax.set_title(title)
        ax.legend()
        
        # Add explanation
        if model == 'random':
            vmr_theo = np.var(theoretical_data) / np.mean(theoretical_data) if np.mean(theoretical_data) > 0 else 0
            vmr_actual = np.var(data) / np.mean(data) if np.mean(data) > 0 else 0
            
            explanation = f"Theoretical log-normal distribution has VMR = {vmr_theo:.2f}\n"
            explanation += f"Empirical data has VMR = {vmr_actual:.2f}\n"
            explanation += "Luria-Delbrück distributions typically have VMR > 1"
            
            ax.text(0.05, 0.05, explanation, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        self.fig.tight_layout()

    def plot_model_comparison(self):
        """Plot comparison of different models"""
        ax = self.fig.add_subplot(111)
        
        # Plot CCDF for each available model
        for model, data in self.simulation_results.items():
            sorted_data = np.sort(data)
            ccdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            if model == 'random':
                color = 'blue'
                label = "Random (Darwinian) Model"
            elif model == 'induced':
                color = 'red'
                label = "Induced (Lamarckian) Model"
            else:
                color = 'green'
                label = "Combined Model"
                
            ax.step(sorted_data, ccdf, where='post', color=color, 
                   label=label, linewidth=2)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Number of Resistant Survivors (log scale)")
        ax.set_ylabel("P(X > x) (log scale)")
        ax.set_title("Model Comparison: Complementary Cumulative Distribution Function")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        
        # Add explanation text
        if 'random' in self.simulation_results and 'induced' in self.simulation_results:
            explanation = "The Random (Darwinian) model typically shows a\nstraighter line than the Induced (Lamarckian) model\non this log-log plot, indicating a power-law-like\ndistribution characteristic of Luria-Delbrück."
            ax.text(0.05, 0.05, explanation, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        self.fig.tight_layout()

    def plot_vmr(self, model):
        """Plot variance-to-mean ratio for a model"""
        ax = self.fig.add_subplot(111)
        
        if model in self.statistics and 'vmr' in self.statistics[model]:
            vmr = self.statistics[model]['vmr']
            
            # Create bar chart
            ax.bar([model.capitalize()], [vmr], color='purple', alpha=0.7)
            ax.set_ylabel('Variance-to-Mean Ratio')
            ax.set_title(f'{model.capitalize()} Model: Variance-to-Mean Ratio')
            
            # Add horizontal line at VMR = 1 (Poisson expectation)
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)
            
            # Add value label
            ax.text(0, vmr, f"{vmr:.2f}", ha='center', va='bottom', fontweight='bold')
            
            # Add explanation
            if vmr > 1:
                explanation = "VMR > 1 supports Luria-Delbrück hypothesis"
            else:
                explanation = "VMR ≈ 1 is expected for Poisson distribution (induced mutations)"
                
            ax.text(0.5, 0.05, explanation, transform=ax.transAxes, fontsize=12,
                    ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        else:
            ax.text(0.5, 0.5, "No VMR statistics available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        self.fig.tight_layout()

    def plot_vmr_comparison(self):
        """Plot comparison of VMR values for different models"""
        ax = self.fig.add_subplot(111)
        
        # Collect VMR values for each model
        models = []
        vmr_values = []
        colors = []
        
        for model in self.statistics:
            if 'vmr' in self.statistics[model]:
                models.append(model.capitalize())
                vmr_values.append(self.statistics[model]['vmr'])
                
                if model == 'random':
                    colors.append('blue')
                elif model == 'induced':
                    colors.append('red')
                else:
                    colors.append('green')
        
        if models and vmr_values:
            # Create bar chart
            bars = ax.bar(models, vmr_values, color=colors, alpha=0.7)
            ax.set_ylabel('Variance-to-Mean Ratio (VMR)')
            ax.set_title('Variance-to-Mean Ratio Comparison Between Models')
            
            # Add horizontal line at VMR = 1 (Poisson expectation)
            ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, 
                      label='VMR = 1 (Poisson distribution)')
            
            # Add value labels
            for bar, value in zip(bars, vmr_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f"{value:.2f}", ha='center', va='bottom', fontweight='bold')
            
            # Add legend
            ax.legend()
            
            # Add explanation
            explanation = "VMR > 1 supports Luria-Delbrück hypothesis (random mutations)\n"
            explanation += "VMR ≈ 1 is expected for Poisson distribution (induced mutations)"
            
            ax.text(0.5, 0.05, explanation, transform=ax.transAxes, fontsize=10,
                    ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            
            # If random VMR > induced VMR, highlight this
            if 'random' in self.statistics and 'induced' in self.statistics:
                if self.statistics['random']['vmr'] > self.statistics['induced']['vmr']:
                    conclusion = "Random model VMR > Induced model VMR\n"
                    conclusion += "This supports the Luria-Delbrück hypothesis"
                    
                    ax.text(0.5, 0.2, conclusion, transform=ax.transAxes, fontsize=12,
                            ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        else:
            ax.text(0.5, 0.5, "No VMR statistics available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        self.fig.tight_layout()

    def save_results(self):
        """Save results text to a file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, "w") as f:
                f.write(self.results_text.get(1.0, tk.END))
            self.status_var.set(f"Results saved to {file_path}")

    def save_visualization(self):
        """Save current visualization to a file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            self.status_var.set(f"Visualization saved to {file_path}")

    def clear_results(self):
        """Clear results text area"""
        self.results_text.delete(1.0, tk.END)

    def load_results_dir(self):
        """Load results from a previously run simulation"""
        directory = filedialog.askdirectory(
            title="Select Luria-Delbrück Results Directory"
        )
        if directory and os.path.isdir(directory):
            self.current_results_dir = directory
            self.status_var.set(f"Loading results from {directory}")
            self.load_simulation_results(directory)
            
            # Also try to load log file for display
            log_file = os.path.join(directory, "Assignment_Introduction_Luria_Delbrook.log")
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r") as f:
                        content = f.read()
                        self.results_text.delete(1.0, tk.END)
                        self.results_text.insert(tk.END, f"Loaded log from {log_file}:\n\n")
                        self.results_text.insert(tk.END, content)
                except Exception as e:
                    self.results_text.insert(tk.END, f"Error loading log file: {str(e)}\n", "error")
                    self.results_text.tag_configure("error", foreground="red")
            else:
                # Try to load report if available
                self.load_detailed_report()

    def load_detailed_report(self):
        """Load and display the detailed Luria-Delbrück report"""
        if not self.current_results_dir:
            self.status_var.set("No results directory loaded")
            return
            
        report_file = os.path.join(self.current_results_dir, "luria_delbrueck_detailed_report.txt")
        if os.path.exists(report_file):
            try:
                with open(report_file, "r") as f:
                    content = f.read()
                    self.results_text.delete(1.0, tk.END)
                    self.results_text.insert(tk.END, f"Loaded detailed report:\n\n")
                    self.results_text.insert(tk.END, content)
                    self.status_var.set("Loaded detailed report")
            except Exception as e:
                self.results_text.insert(tk.END, f"Error loading report file: {str(e)}\n", "error")
                self.results_text.tag_configure("error", foreground="red")
        else:
            # Try model comparison report instead
            report_file = os.path.join(self.current_results_dir, "model_comparison_report.txt")
            if os.path.exists(report_file):
                try:
                    with open(report_file, "r") as f:
                        content = f.read()
                        self.results_text.delete(1.0, tk.END)
                        self.results_text.insert(tk.END, f"Loaded model comparison report:\n\n")
                        self.results_text.insert(tk.END, content)
                        self.status_var.set("Loaded model comparison report")
                except Exception as e:
                    self.results_text.insert(tk.END, f"Error loading report file: {str(e)}\n", "error")
                    self.results_text.tag_configure("error", foreground="red")
            else:
                self.status_var.set("No report files found in the results directory")
                self.results_text.insert(tk.END, "No report files found in the results directory\n")

    def open_results_folder(self):
        """Open the results folder in file explorer"""
        if self.current_results_dir and os.path.isdir(self.current_results_dir):
            try:
                if sys.platform == 'win32':
                    os.startfile(self.current_results_dir)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.call(['open', self.current_results_dir])
                else:  # Linux
                    subprocess.call(['xdg-open', self.current_results_dir])
                self.status_var.set(f"Opened results folder: {self.current_results_dir}")
            except Exception as e:
                self.status_var.set(f"Error opening folder: {str(e)}")
        else:
            self.status_var.set("No results directory available")


def main():
    root = tk.Tk()
    app = LuriaDelbrueckGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()