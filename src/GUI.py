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
            'Theoretical Comparison'
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
        
        # Save visualization button
        ttk.Button(control_frame, text="Save Image", command=self.save_visualization).pack(side=tk.RIGHT, padx=5)

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
        else:
            # Restore previous values if they exist
            if hasattr(self, 'stored_values'):
                self.cultures_var.set(self.stored_values['cultures'])
                self.pop_size_var.set(self.stored_values['pop_size'])
                self.generations_var.set(self.stored_values['generations'])
                self.mut_rate_var.set(self.stored_values['mut_rate'])
                self.induced_rate_var.set(self.stored_values['induced_rate'])

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
            
            # Read and display output in real-time
            for line in iter(process.stdout.readline, ''):
                self.queue.put(('output', line))
            
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
                elif msg_type == 'complete':
                    # Re-enable UI elements
                    self.run_button.config(state="normal")
                    self.progress.stop()
                    
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
            self.simulation_results = {}
            
            # Look for CSV files with survivor data
            for model in ['random', 'induced', 'combined']:
                csv_path = os.path.join(directory, f"survivors_{model}.csv")
                if os.path.exists(csv_path):
                    data = np.loadtxt(csv_path, delimiter=',')
                    self.simulation_results[model] = data
            
            # If we found any results, update visualization
            if self.simulation_results:
                self.queue.put(('status', f"Loaded results for {', '.join(self.simulation_results.keys())}"))
                self.update_visualization_options()
                
                # Set default visualization to first available model
                if self.simulation_results and self.vis_model_var.get() not in self.simulation_results:
                    self.vis_model_var.set(next(iter(self.simulation_results.keys())))
                
                # Update visualization
                self.update_visualization(None)
        except Exception as e:
            self.queue.put(('error', f"Error loading results: {str(e)}"))

    def update_visualization_options(self):
        """Update the visualization options based on available results"""
        if self.simulation_results:
            # Update model selection combobox
            available_models = list(self.simulation_results.keys())
            self.vis_model_combo['values'] = available_models
            
            # If multiple models are available, enable comparison options
            if len(available_models) > 1:
                all_values = list(self.vis_type_var['values'])
                if 'Model Comparison' not in all_values:
                    new_values = list(all_values) + ['Model Comparison']
                    self.vis_type_var['values'] = new_values

    def update_visualization(self, event):
        """Update the visualization based on selected type and model"""
        if not self.simulation_results:
            return
            
        vis_type = self.vis_type_var.get()
        model = self.vis_model_var.get()
        
        # Clear the figure
        self.fig.clear()
        
        # Get data for the selected model
        if model in self.simulation_results:
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
            elif vis_type == 'Model Comparison' and len(self.simulation_results) > 1:
                self.plot_model_comparison()
        
        # Refresh the canvas
        self.canvas.draw()

    def plot_standard_distribution(self, data, model):
        """Plot standard histogram of survivor counts"""
        ax = self.fig.add_subplot(111)
        ax.hist(data, bins='auto', alpha=0.7, color='royalblue')
        ax.set_xlabel("Number of Resistant Survivors")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{model.capitalize()} Model: Distribution of Resistant Survivors")
        self.fig.tight_layout()

    def plot_log_scale_distribution(self, data, model):
        """Plot log-scale histogram of survivor counts"""
        ax = self.fig.add_subplot(111)
        
        if max(data) > 0:
            # Add a small value to handle zeros when using log scale
            nonzero_data = [x + 0.1 for x in data]
            bins = np.logspace(np.log10(0.1), np.log10(max(nonzero_data)), 20)
            ax.hist(nonzero_data, bins=bins, alpha=0.7, color='forestgreen')
            ax.set_xscale('log')
            ax.set_xlabel("Number of Resistant Survivors (log scale)")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{model.capitalize()} Model: Log-Scale Distribution")
        else:
            ax.text(0.5, 0.5, "No data available for log scale", 
                   ha='center', va='center', transform=ax.transAxes)
        
        self.fig.tight_layout()

    def plot_ccdf(self, data, model):
        """Plot complementary cumulative distribution function"""
        ax = self.fig.add_subplot(111)
        
        sorted_data = np.sort(data)
        ccdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        ax.step(sorted_data, ccdf, where='post', color='darkorange')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Number of Resistant Survivors (log scale)")
        ax.set_ylabel("P(X > x) (log scale)")
        ax.set_title(f"{model.capitalize()} Model: CCDF")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        self.fig.tight_layout()

    def plot_theoretical_comparison(self, data, model):
        """Plot comparison with theoretical distribution"""
        ax = self.fig.add_subplot(111)
        
        # Generate theoretical data for comparison
        if model == "random":
            # Simulate Luria-Delbrück distribution (simplified)
            # Using log-normal as an approximation
            theoretical_data = np.random.lognormal(0, 1, size=1000)
            title = "Random Model vs. Theoretical Luria-Delbrück Distribution"
        else:
            # Simulate Poisson distribution for induced model
            theoretical_data = np.random.poisson(np.mean(data), size=1000)
            title = f"{model.capitalize()} Model vs. Theoretical Distribution"
        
        # Use kernel density estimation for smooth curves
        x = np.linspace(0, max(max(data), max(theoretical_data)) * 1.1, 1000)
        
        # Calculate KDE for simulation data
        from scipy.stats import gaussian_kde
        try:
            kde_data = gaussian_kde(data)
            y_data = kde_data(x)
            ax.plot(x, y_data, color='blue', label="Simulation data")
        except:
            # Fallback if KDE fails
            ax.hist(data, bins='auto', alpha=0.3, color='blue', density=True, label="Simulation data")
        
        # Calculate KDE for theoretical data
        try:
            kde_theory = gaussian_kde(theoretical_data)
            y_theory = kde_theory(x)
            ax.plot(x, y_theory, color='red', label="Theoretical distribution")
        except:
            # Fallback if KDE fails
            ax.hist(theoretical_data, bins='auto', alpha=0.3, color='red', density=True, label="Theoretical")
        
        ax.set_xlabel("Number of Resistant Survivors")
        ax.set_ylabel("Probability Density")
        ax.set_title(title)
        ax.legend()
        
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
            elif model == 'induced':
                color = 'red'
            else:
                color = 'green'
                
            ax.step(sorted_data, ccdf, where='post', color=color, 
                   label=f"{model.capitalize()} Model")
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Number of Resistant Survivors (log scale)")
        ax.set_ylabel("P(X > x) (log scale)")
        ax.set_title("Model Comparison: CCDF")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        
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
                    self.results_text.insert(tk.END, f"Error loading log file: {str(e)}\n")
            
            # Try to load report if available
            report_file = os.path.join(directory, "luria_delbrueck_detailed_report.txt")
            if os.path.exists(report_file):
                try:
                    with open(report_file, "r") as f:
                        content = f.read()
                        self.results_text.delete(1.0, tk.END)
                        self.results_text.insert(tk.END, f"Loaded detailed report:\n\n")
                        self.results_text.insert(tk.END, content)
                except Exception as e:
                    self.results_text.insert(tk.END, f"Error loading report file: {str(e)}\n")


def main():
    root = tk.Tk()
    app = LuriaDelbrueckGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()