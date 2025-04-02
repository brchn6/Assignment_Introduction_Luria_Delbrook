import numpy as np
import uuid
import argparse 
import os
import sys
import logging as log
from tqdm import tqdm
from datetime import datetime
import shutil
import time 
import cProfile
import pstats
import multiprocessing as mp
from functools import partial
import pandas as pd

# Helper function needed for the improved visualization
from mpl_toolkits.axes_grid1 import make_axes_locatable

#######################################################################
# Setup logging
#######################################################################
def init_log(results_path, log_level="INFO"):
    """Initialize logging with specified log level"""
    # Set log level
    level = getattr(log, log_level.upper(), log.INFO)

    # Initialize logging
    log.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    
    # Add file handler
    path = os.path.join(results_path, 'Assignment_Introduction_Luria_Delbrook.log')
    fh = log.FileHandler(path, mode='w')
    fh.setLevel(level)
    log.getLogger().addHandler(fh)

    log.info("Logging initialized successfully.")
    log.info(f"Results will be saved to: {results_path}")

    return log, results_path

def _get_lsf_job_details() -> list[str]:
    """
    Retrieves environment variables for LSF job details, if available.
    """
    lsf_job_id = os.environ.get('LSB_JOBID')    # Job ID
    lsf_queue = os.environ.get('LSB_QUEUE')     # Queue name
    lsf_host = os.environ.get('LSB_HOSTS')      # Hosts allocated
    lsf_job_name = os.environ.get('LSB_JOBNAME')  # Job name
    lsf_command = os.environ.get('LSB_CMD')     # Command used to submit job

    details = [
        f"LSF Job ID: {lsf_job_id}",
        f"LSF Queue: {lsf_queue}",
        f"LSF Hosts: {lsf_host}",
        f"LSF Job Name: {lsf_job_name}",
        f"LSF Command: {lsf_command}"
    ]
    return details

#######################################################################
# Time tracking decorator
#######################################################################
def timing_decorator(func):
    """Decorator to measure and log execution time of functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        log.info(f"Function {func.__name__} took {elapsed_time:.2f} seconds to execute")
        return result, elapsed_time
    return wrapper

#######################################################################
# Classes
#######################################################################
class Organism:
    N_GENES = 1  # Default value, will be overridden by command line arg
    P_MUTATION = 0.1  # Default, overridden in main
    
    def __init__(self, genome=None, id=None):
        if genome is not None:
            self.genome = genome
        else:
            # Initialize with non-resistant genome (all zeros)
            self.genome = np.zeros(Organism.N_GENES, dtype=int)
        
        # Only generate UUID if explicitly needed to reduce overhead
        self.id = id if id is not None else None
        
        # For tracking if this organism is resistant (first gene determines resistance)
        self.is_resistant = bool(self.genome[0])

    def __repr__(self):
        if self.id is None:
            self.id = uuid.uuid4()
        status = "Resistant" if self.is_resistant else "Sensitive"
        return f"Organism {self.id}: {status}"

    def mutate(self, mutation_rate=None):
        """Apply random mutations to the genome based on mutation probability"""
        rate = mutation_rate if mutation_rate is not None else Organism.P_MUTATION
        
        # Check for mutations in each gene
        for i in range(Organism.N_GENES):
            if np.random.rand() < rate:
                self.genome[i] = 1 - self.genome[i]  # Flip bit
        
        # Update resistance status (always determined by first gene)
        self.is_resistant = bool(self.genome[0])
        return self

    def reproduce(self, mutation_rate=None):
        """Create an offspring with possible mutations"""
        offspring = Organism(genome=self.genome.copy())
        if mutation_rate is not None:
            offspring.mutate(mutation_rate)
        else:
            offspring.mutate()
        return offspring

    @property
    def resistance_gene(self):
        """Get the value of the resistance gene"""
        return self.genome[0]
        
    @property
    def gene_count(self):
        """Get number of genes that are turned on (value=1)"""
        return np.sum(self.genome)

#######################################################################
# Simulation functions 
#######################################################################
def process_culture(args):
    """Generic process function for all models - designed for parallel processing"""
    culture_id, model_type, initial_population_size, num_generations, mutation_rate, induced_mutation_rate, log_first_cultures = args
    
    # Initialize population with sensitive organisms
    population = [Organism() for _ in range(initial_population_size)]
    
    # Track generation stats for this culture if it's the first one and logging is enabled
    generation_stats = [] if culture_id == 0 and log_first_cultures else None
    
    # Different behavior based on model type
    if model_type == "random":
        # Simulate generations with random mutations
        for gen in range(num_generations):
            new_population = []
            for organism in population:
                # Each organism produces two offspring with possible mutations
                offspring1 = organism.reproduce(mutation_rate)
                offspring2 = organism.reproduce(mutation_rate)
                
                new_population.append(offspring1)
                new_population.append(offspring2)
                
            population = new_population
            
            # Track statistics for this generation if needed
            if generation_stats is not None:
                resistant_count = sum(org.is_resistant for org in population)
                resistant_percent = (resistant_count / len(population)) * 100
                generation_stats.append({
                    'gen': gen + 1,
                    'population': len(population),
                    'resistant': resistant_count,
                    'resistant_percent': resistant_percent
                })
        
        # For random model, no further mutations after growth
        pre_induced_resistant = sum(org.is_resistant for org in population)
        survivors = pre_induced_resistant
        
    elif model_type == "induced":
        # Simulate generations with no mutations during growth
        for gen in range(num_generations):
            new_population = []
            for organism in population:
                # Each organism reproduces without mutation
                offspring1 = Organism(genome=organism.genome.copy())
                offspring2 = Organism(genome=organism.genome.copy())
                
                new_population.append(offspring1)
                new_population.append(offspring2)
                
            population = new_population
            
            # Track statistics for this generation if needed
            if generation_stats is not None:
                resistant_count = sum(org.is_resistant for org in population)
                resistant_percent = (resistant_count / len(population)) * 100
                generation_stats.append({
                    'gen': gen + 1,
                    'population': len(population),
                    'resistant': resistant_count,
                    'resistant_percent': resistant_percent
                })
        
        # Apply induced mutations at the end
        pre_induced_resistant = 0  # Always 0 for induced model
        for organism in population:
            organism.mutate(mutation_rate)  # Apply with specified rate
            
        survivors = sum(org.is_resistant for org in population)
        
    else:  # combined model
        # Simulate generations with random mutations
        for gen in range(num_generations):
            new_population = []
            for organism in population:
                # Each organism produces two offspring with possible mutations
                offspring1 = organism.reproduce(mutation_rate)
                offspring2 = organism.reproduce(mutation_rate)
                
                new_population.append(offspring1)
                new_population.append(offspring2)
                
            population = new_population
            
            # Track statistics for this generation if needed
            if generation_stats is not None:
                resistant_count = sum(org.is_resistant for org in population)
                resistant_percent = (resistant_count / len(population)) * 100
                generation_stats.append({
                    'gen': gen + 1,
                    'population': len(population),
                    'resistant': resistant_count,
                    'resistant_percent': resistant_percent
                })
        
        # Store pre-induced mutation stats
        pre_induced_resistant = sum(org.is_resistant for org in population)
        
        # Apply additional induced mutations at the end
        for organism in population:
            organism.mutate(induced_mutation_rate)  # Apply with induced rate
            
        survivors = sum(org.is_resistant for org in population)
    
    # Return results with appropriate format based on model
    if model_type == "combined":
        return survivors, generation_stats, pre_induced_resistant
    else:
        return survivors, generation_stats

@timing_decorator
def simulate_mutations(model_type, num_cultures=1000, initial_population_size=10, num_generations=5, 
                     mutation_rate=0.1, induced_mutation_rate=0.1, use_parallel=True, n_processes=None):
    """
    Generic simulation function for all mutation models.
    Returns a list of 'survivors' (resistant cell counts) across cultures.
    """
    # Set the class variables for all Organisms
    Organism.P_MUTATION = mutation_rate if model_type != "induced" else 0
    
    survivors_list = []
    generation_stats = None
    pre_induced_resistant = 0 if model_type == "combined" else None
    
    log.debug(f"Starting {model_type} mutation simulation with {num_cultures} cultures")

    # Prepare arguments for each culture
    culture_args = [(culture_id, model_type, initial_population_size, num_generations, 
                    mutation_rate, induced_mutation_rate, True if culture_id == 0 else False) 
                    for culture_id in range(num_cultures)]

    if use_parallel and num_cultures > 10:
        # Use multiprocessing for better performance
        n_processes = n_processes or max(1, mp.cpu_count() - 1)
        log.info(f"Using parallel processing with {n_processes} processes")
        
        # Process cultures in parallel
        with mp.Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap(process_culture, culture_args),
                total=num_cultures,
                desc=f"{model_type.capitalize()} Model Progress",
                unit="culture"
            ))
        
        # Process results based on model type
        if model_type == "combined":
            for i, (survivors, stats, pre_induced) in enumerate(results):
                survivors_list.append(survivors)
                if stats is not None:
                    generation_stats = stats
                if i == 0:
                    pre_induced_resistant = pre_induced
        else:
            for survivors, stats in results:
                survivors_list.append(survivors)
                if stats is not None:
                    generation_stats = stats
        
    else:
        # Sequential processing with progress bar
        for args in tqdm(culture_args, desc=f"{model_type.capitalize()} Model Progress", unit="culture"):
            if model_type == "combined":
                survivors, stats, pre_induced = process_culture(args)
                survivors_list.append(survivors)
                if stats is not None:
                    generation_stats = stats
                if args[0] == 0:  # First culture
                    pre_induced_resistant = pre_induced
            else:
                survivors, stats = process_culture(args)
                survivors_list.append(survivors)
                if stats is not None:
                    generation_stats = stats
    
    # Log generation progression for first culture
    if generation_stats:
        log.info("Generation progression (first culture):")
        for stat in generation_stats:
            log.info(f"  Generation {stat['gen']}: {stat['resistant']} resistant ({stat['resistant_percent']:.2f}%)")
        
        # Additional logging for combined model
        if model_type == "combined" and pre_induced_resistant is not None:
            post_induced_resistant = survivors_list[0]
            induced_effect = post_induced_resistant - pre_induced_resistant
            log.info(f"  Before induced mutations: {pre_induced_resistant} resistant")
            log.info(f"  After induced mutations: {post_induced_resistant} resistant (added {induced_effect})")

    return survivors_list

#######################################################################
# Statistical Analysis Functions
#######################################################################
@timing_decorator
def analyze_results(survivors_list, model_name):
    """Perform statistical analysis on the simulation results."""
    mean_survivors = np.mean(survivors_list)
    var_survivors = np.var(survivors_list)
    median_survivors = np.median(survivors_list)
    max_survivors = np.max(survivors_list)
    
    # Calculate coefficient of variation (std/mean) - important for distinguishing models
    std_survivors = np.std(survivors_list)
    cv = std_survivors / mean_survivors if mean_survivors > 0 else 0
    
    # Calculate variance-to-mean ratio (VMR) - key for Luria-Delbrück analysis
    vmr = var_survivors / mean_survivors if mean_survivors > 0 else 0
    
    # Count cultures with zero resistant organisms
    zero_resistant_count = sum(1 for s in survivors_list if s == 0)
    zero_resistant_percent = (zero_resistant_count / len(survivors_list)) * 100
    
    results = {
        'model': model_name,
        'mean': mean_survivors,
        'variance': var_survivors,
        'std_dev': std_survivors,
        'median': median_survivors,
        'max': max_survivors,
        'coefficient_of_variation': cv,
        'variance_to_mean_ratio': vmr,
        'zero_resistant_cultures': zero_resistant_count,
        'zero_resistant_percent': zero_resistant_percent
    }
    
    # If we're tracking multiple genes, additional analysis could be added here
    if Organism.N_GENES > 1:
        log.info(f"Multi-gene analysis: Tracking {Organism.N_GENES} genes per organism")
        # Additional multi-gene analysis could be implemented here
    
    return results

#######################################################################
# Visualization Functions
#######################################################################
@timing_decorator
def create_visualizations(survivors_list, model_name, results_path):
    """Create visualizations of the simulation results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style for better visualizations
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 12))
        
        # 1. Standard histogram (top left)
        plt.subplot(2, 2, 1)
        plt.hist(survivors_list, bins='auto', alpha=0.7, color='royalblue')
        plt.xlabel("Number of Resistant Survivors")
        plt.ylabel("Frequency")
        plt.title(f"{model_name.capitalize()} Model: Distribution of Resistant Survivors")
        
        # 2. Log-scale histogram - KEY for Luria-Delbrück (top right)
        plt.subplot(2, 2, 2)
        if max(survivors_list) > 0:
            # Add a small value to handle zeros when using log scale
            nonzero_data = [x + 0.1 for x in survivors_list if x > 0]
            if len(nonzero_data) > 0:
                bins = np.logspace(np.log10(min(nonzero_data)), np.log10(max(nonzero_data)), 20)
                plt.hist(nonzero_data, bins=bins, alpha=0.7, color='forestgreen')
                plt.xscale('log')
                plt.xlabel("Number of Resistant Survivors (log scale)")
                plt.ylabel("Frequency")
                plt.title("Log-Scale Distribution")
        
        # 3. CCDF plot (bottom left) - Excellent for showing power-law type distributions
        plt.subplot(2, 2, 3)
        sorted_data = np.sort(survivors_list)
        ccdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.step(sorted_data, ccdf, where='post', color='darkorange')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Number of Resistant Survivors (log scale)")
        plt.ylabel("P(X > x) (log scale)")
        plt.title("Complementary Cumulative Distribution Function")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # 4. Probability density function with fit (bottom right)
        plt.subplot(2, 2, 4)
        sns.kdeplot(survivors_list, color='purple', label="Empirical distribution")
        plt.xlabel("Number of Resistant Survivors")
        plt.ylabel("Probability Density")
        plt.title("Probability Density Function")
        
        plt.tight_layout()
        save_path = os.path.join(results_path, f"luria_delbrueck_analysis_{model_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        # Create a second figure specifically for model comparison if comparing random vs induced
        if model_name in ["random", "induced"]:
            # Create a special figure showing theoretical distributions
            plt.figure(figsize=(10, 6))
            
            # Generate theoretical data for comparison
            if model_name == "random":
                # Simulate Luria-Delbrück distribution (simplified)
                # Using log-normal as an approximation
                nonzero_data = [x for x in survivors_list if x > 0]
                if len(nonzero_data) > 0:
                    log_data = np.log(nonzero_data)
                    mean_log = np.mean(log_data)
                    sigma_log = np.std(log_data)
                    theoretical_data = np.random.lognormal(mean_log, sigma_log, size=10000)
                    plt.title("Random Mutation Model vs. Theoretical Luria-Delbrück Distribution")
            else:
                # Simulate Poisson distribution for induced model
                theoretical_data = np.random.poisson(np.mean(survivors_list), size=1000)
                plt.title("Induced Mutation Model vs. Theoretical Poisson Distribution")
            
            # Plot both empirical and theoretical distributions
            sns.kdeplot(survivors_list, color='blue', label="Simulation data")
            sns.kdeplot(theoretical_data, color='red', label="Theoretical distribution")
            plt.xlabel("Number of Resistant Survivors")
            plt.ylabel("Probability Density")
            plt.legend()
            
            theory_path = os.path.join(results_path, f"theoretical_comparison_{model_name}.png")
            plt.savefig(theory_path, dpi=300)
            plt.close()
        
        return save_path
        
    except ImportError as e:
        log.warning(f"Visualization libraries not available: {e}")
        return None

@timing_decorator
def create_comparison_visualization(results_dict, results_path):
    """Create comparison visualizations for all models."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.gridspec import GridSpec
        
        # Set style for better visualizations
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # Get model names and their corresponding data
        model_names = list(results_dict.keys())
        
        # Create figure with custom grid layout
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. Distribution comparison with separate y-axes (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        
        colors = {'random': 'royalblue', 'induced': 'firebrick', 'combined': 'forestgreen'}
        model_display_names = {'random': 'Random', 'induced': 'Induced', 'combined': 'Combined'}
        
        # Use kernel density estimation with different scales for each model
        for i, model in enumerate(model_names):
            # Create a twin y-axis for each additional model
            if i == 0:
                curr_ax = ax1
            else:
                curr_ax = ax1.twinx()
                # Offset the right spine for multiple twin axes
                if i > 1:
                    curr_ax.spines['right'].set_position(('outward', 60 * (i-1)))
            
            # Plot the density curve
            sns.kdeplot(results_dict[model], ax=curr_ax, color=colors[model], 
                       label=model_display_names[model], linewidth=2.5)
            
            # Set y-label with matching color
            curr_ax.set_ylabel(f"{model_display_names[model]} Density", color=colors[model])
            curr_ax.tick_params(axis='y', colors=colors[model])
        
        # Set common x label
        ax1.set_xlabel("Number of Resistant Survivors")
        ax1.set_title("Distribution Comparison (Separate Scales)", fontweight='bold')
        
        # Add legends for all models
        lines, labels = [], []
        for ax in [ax1] + ([ax for ax in fig.axes if ax != ax1 and ax.bbox.bounds[1] > 0.5]):
            line, label = ax.get_legend_handles_labels()
            lines.extend(line)
            labels.extend(label)
        ax1.legend(lines, labels, loc='upper right')
        
        # 2. Box plot comparison with log scale (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Prepare data for boxplot
        data_for_boxplot = [results_dict[model] for model in model_names]
        box_colors = [colors[model] for model in model_names]
        
        # Create boxplot with customized appearance
        boxplots = ax2.boxplot(data_for_boxplot, 
                            vert=True, 
                            patch_artist=True,
                            labels=[model_display_names[m] for m in model_names],
                            showfliers=False)  # Hide outliers for cleaner display
        
        # Customize boxplot colors
        for patch, color in zip(boxplots['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Set log scale if range is large
        ranges = [max(data) - min(data) for data in data_for_boxplot]
        if max(ranges) / min(ranges) > 100:
            ax2.set_yscale('log')
            ax2.set_title("Box Plot Comparison (Log Scale)", fontweight='bold')
        else:
            ax2.set_title("Box Plot Comparison", fontweight='bold')
            
        ax2.set_ylabel("Number of Resistant Survivors")
        ax2.grid(True, axis='y', alpha=0.3)
        
        # 3. Variance-to-Mean Ratio Comparison (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Calculate VMR for each model
        vmr_values = []
        for model in model_names:
            survivors = results_dict[model]
            mean = np.mean(survivors)
            var = np.var(survivors)
            vmr = var / mean if mean > 0 else 0
            vmr_values.append(vmr)
        
        # Create bar chart with custom colors
        bars = ax3.bar([model_display_names[m] for m in model_names], vmr_values, 
                       color=[colors[m] for m in model_names], alpha=0.7, edgecolor='black')
        
        # Determine if log scale is needed
        if max(vmr_values) / (min(vmr_values) + 0.001) > 100:
            ax3.set_yscale('log')
            ax3.set_title("Variance-to-Mean Ratio (Log Scale)", fontweight='bold')
        else:
            ax3.set_title("Variance-to-Mean Ratio Comparison", fontweight='bold')
            
        ax3.set_xlabel("Model")
        ax3.set_ylabel("Variance-to-Mean Ratio")
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                if ax3.get_yscale() == 'log':
                    y_pos = height * 1.1
                else:
                    y_pos = height + max(vmr_values) * 0.02
                ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{height:.2f}', ha='center', va='bottom')
        
        # Add Luria-Delbrück reference line and annotation
        ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        ax3.text(0.02, 0.02, "VMR ≈ 1 for Poisson distribution (expected for induced mutations)\nVMR > 1 supports Luria-Delbrück hypothesis", 
                transform=ax3.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 4. Summary statistics table (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create table data
        stats = []
        headers = ["Statistic"] + [model_display_names[m] for m in model_names]
        
        # Calculate coefficient of variation for highlighting
        cv_values = []
        for model in model_names:
            data = results_dict[model]
            mean = np.mean(data)
            std = np.std(data)
            cv = std / mean if mean > 0 else 0
            cv_values.append(cv)
        
        # Add rows of statistics
        stats.append(["Mean"] + [f"{np.mean(results_dict[m]):.2f}" for m in model_names])
        stats.append(["Variance"] + [f"{np.var(results_dict[m]):.2f}" for m in model_names])
        stats.append(["Std Dev"] + [f"{np.std(results_dict[m]):.2f}" for m in model_names])
        stats.append(["Median"] + [f"{np.median(results_dict[m]):.2f}" for m in model_names])
        stats.append(["Max"] + [f"{np.max(results_dict[m])}" for m in model_names])
        stats.append(["CV"] + [f"{cv_values[i]:.2f}" for i in range(len(model_names))])
        stats.append(["VMR"] + [f"{vmr_values[i]:.2f}" for i in range(len(model_names))])
        stats.append(["Zero Count"] + [f"{sum(1 for x in results_dict[m] if x == 0)}" for m in model_names])
        
        # Create the table with custom coloring
        table = ax4.table(cellText=stats, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        
        # Highlight cells for better readability
        for i in range(len(stats)):
            for j in range(len(model_names) + 1):
                cell = table[(i+1, j)]
                
                # Shade header cells
                if i == -1 or j == 0:
                    cell.set_facecolor('#D8D8D8')
                    cell.set_text_props(weight='bold')
                
                # Highlight the model with highest CV (key for Luria-Delbrück)
                if i == 5 and j > 0:  # CV row
                    model_idx = j - 1
                    if cv_values[model_idx] == max(cv_values):
                        cell.set_facecolor('#BBFFBB')
                
                # Highlight VMR values greater than 1 (supporting Luria-Delbrück)
                if i == 6 and j > 0:  # VMR row
                    model_idx = j - 1
                    if vmr_values[model_idx] > 1:
                        cell.set_facecolor('#BBFFBB')
                    elif vmr_values[model_idx] < 1.1 and vmr_values[model_idx] > 0.9:
                        cell.set_facecolor('#FFFFBB')  # Yellow for values close to 1 (Poisson-like)
        
        # Add a title for the table
        ax4.set_title("Summary Statistics", fontweight='bold')
        
        # Add an explanation of the key findings
        if 'random' in model_names and 'induced' in model_names:
            random_idx = model_names.index('random')
            induced_idx = model_names.index('induced')
            
            if vmr_values[random_idx] > vmr_values[induced_idx]:
                conclusion = "The Random model shows higher variance-to-mean ratio than the Induced model, supporting the Luria-Delbrück hypothesis."
            else:
                conclusion = "Results are inconclusive: expected the Random model to show higher variance-to-mean ratio than the Induced model."
                
            ax4.text(0.5, 0.02, conclusion, transform=ax4.transAxes, 
                    fontsize=11, ha='center', va='bottom', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(results_path, "model_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Create a dedicated histogram comparison figure
        plt.figure(figsize=(15, 10))
        
        # Use facet grid for clearer multi-model histogram comparison
        if len(model_names) > 1:
            # Prepare data for facet grid
            all_data = []
            for model in model_names:
                model_data = results_dict[model]
                model_df = pd.DataFrame({
                    'survivors': model_data,
                    'model': model_display_names[model]
                })
                all_data.append(model_df)
            
            combined_df = pd.concat(all_data)
            
            # Create facet grid of histograms
            g = sns.FacetGrid(combined_df, col="model", height=4, aspect=1.2,
                             sharex=False, sharey=False)
            g.map_dataframe(sns.histplot, x="survivors", kde=True)
            g.set_axis_labels("Number of Resistant Survivors", "Frequency")
            g.set_titles("{col} Model")
            g.tight_layout()
            
            # Save histogram comparison
            hist_path = os.path.join(results_path, "histogram_comparison.png")
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        
        plt.close('all')
        
        return save_path
    except ImportError as e:
        log.warning(f"Visualization libraries not available: {e}")
        return None
    except Exception as e:
        log.error(f"Error creating comparison visualizations: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        return None

#######################################################################
# Run all models and compare
#######################################################################
@timing_decorator
def run_all_models(args):
    """Run all three models and create comparison visualizations."""
    log.info("Running all three models for comparison...")
    
    results = {}
    stats = {}
    elapsed_times = {}
    
    # 1. Run Random Mutations Model
    log.info("\n" + "="*50)
    log.info("Running Random Mutations Model")
    log.info("="*50)
    
    Organism.N_GENES = args.n_genes
    
    random_survivors, random_time = simulate_mutations(
        model_type="random",
        num_cultures=args.num_cultures,
        initial_population_size=args.initial_population_size,
        num_generations=args.num_generations,
        mutation_rate=args.mutation_rate,
        use_parallel=args.use_parallel,
        n_processes=args.n_processes
    )
    
    results['random'] = random_survivors
    elapsed_times['random'] = random_time
    stats_result, stats_time = analyze_results(random_survivors, 'random')
    stats['random'] = stats_result
    elapsed_times['random_analysis'] = stats_time
    
    # Visualizations
    vis_path, vis_time = create_visualizations(random_survivors, 'random', args.results_path)
    elapsed_times['random_visualization'] = vis_time
    
    # 2. Run Induced Mutations Model
    log.info("\n" + "="*50)
    log.info("Running Induced Mutations Model")
    log.info("="*50)
    
    induced_survivors, induced_time = simulate_mutations(
        model_type="induced",
        num_cultures=args.num_cultures,
        initial_population_size=args.initial_population_size,
        num_generations=args.num_generations,
        mutation_rate=args.mutation_rate,
        use_parallel=args.use_parallel,
        n_processes=args.n_processes
    )
    
    results['induced'] = induced_survivors
    elapsed_times['induced'] = induced_time
    stats_result, stats_time = analyze_results(induced_survivors, 'induced')
    stats['induced'] = stats_result
    elapsed_times['induced_analysis'] = stats_time
    
    # Visualizations
    vis_path, vis_time = create_visualizations(induced_survivors, 'induced', args.results_path)
    elapsed_times['induced_visualization'] = vis_time
    
    # 3. Run Combined Mutations Model
    log.info("\n" + "="*50)
    log.info("Running Combined Mutations Model")
    log.info("="*50)
    
    combined_survivors, combined_time = simulate_mutations(
        model_type="combined",
        num_cultures=args.num_cultures,
        initial_population_size=args.initial_population_size,
        num_generations=args.num_generations,
        mutation_rate=args.mutation_rate,
        induced_mutation_rate=args.induced_mutation_rate,
        use_parallel=args.use_parallel,
        n_processes=args.n_processes
    )
    
    results['combined'] = combined_survivors
    elapsed_times['combined'] = combined_time
    stats_result, stats_time = analyze_results(combined_survivors, 'combined')
    stats['combined'] = stats_result
    elapsed_times['combined_analysis'] = stats_time
    
    # Visualizations
    vis_path, vis_time = create_visualizations(combined_survivors, 'combined', args.results_path)
    elapsed_times['combined_visualization'] = vis_time
    
    # 4. Create comparison visualizations
    log.info("\n" + "="*50)
    log.info("Creating Comparison Visualizations")
    log.info("="*50)
    
    comp_path, comparison_time = create_comparison_visualization(results, args.results_path)
    elapsed_times['comparison_visualization'] = comparison_time
    
    # 5. Create comparison report
    report_time_start = time.time()
    create_comparison_report(stats, args.results_path, elapsed_times)
    report_time = time.time() - report_time_start
    elapsed_times['report_creation'] = report_time
    
    # 6. Create execution time summary
    create_time_summary(elapsed_times, args.results_path)
    
    log.info("\n" + "="*50)
    log.info("All models completed. Comparison analysis available at:")
    log.info(f"  {args.results_path}")
    log.info("="*50)
    
    total_time = sum(elapsed_times.values())
    log.info(f"Total execution time: {total_time:.2f} seconds")
    
    return results, stats, elapsed_times

def create_comparison_report(stats, results_path, elapsed_times=None):
    """Create a text report comparing the statistics of all models."""
    report_path = os.path.join(results_path, "model_comparison_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Luria-Delbrück Experiment: Model Comparison Report\n")
        f.write("="*80 + "\n\n")
        
        # Write introduction
        f.write("This report compares three models of mutation in the Luria-Delbrück experiment:\n")
        f.write("1. Random (Darwinian) Model: Mutations occur spontaneously during growth\n")
        f.write("2. Induced (Lamarckian) Model: Mutations occur only in response to selective pressure\n")
        f.write("3. Combined Model: Both mechanisms operate simultaneously\n\n")
        
        # Add information about the number of genes being tracked
        f.write(f"Simulation Configuration:\n")
        f.write(f"- Number of genes per organism: {Organism.N_GENES}\n")
        f.write(f"- First gene determines resistance status\n")
        if Organism.N_GENES > 1:
            f.write(f"- Additional genes ({Organism.N_GENES-1}) are tracked but do not affect resistance\n")
        f.write("\n")

        # Get coefficient of variation for all models
        cv_random = stats['random']['coefficient_of_variation']
        cv_induced = stats['induced']['coefficient_of_variation']
        cv_combined = stats['combined']['coefficient_of_variation']
        
        # Calculate variance-to-mean ratios
        vmr_random = stats['random']['variance_to_mean_ratio'] if 'variance_to_mean_ratio' in stats['random'] else stats['random']['variance'] / stats['random']['mean']
        vmr_induced = stats['induced']['variance_to_mean_ratio'] if 'variance_to_mean_ratio' in stats['induced'] else stats['induced']['variance'] / stats['induced']['mean']
        vmr_combined = stats['combined']['variance_to_mean_ratio'] if 'variance_to_mean_ratio' in stats['combined'] else stats['combined']['variance'] / stats['combined']['mean']
        
        # Determine key findings
        if vmr_random > vmr_induced:
            f.write(f"- The Random model shows higher variance-to-mean ratio (VMR = {vmr_random:.2f}) than the Induced model ")
            f.write(f"(VMR = {vmr_induced:.2f}), consistent with Luria & Delbrück's findings supporting Darwinian evolution.\n")
        else:
            f.write(f"- NOTE: Unexpected result - the Induced model shows higher VMR than Random model.\n")
            f.write(f"  This may be due to parameter choices or statistical fluctuation in the simulation.\n")
        
        f.write(f"- The Combined model (VMR = {vmr_combined:.2f}) exhibits characteristics of both mechanisms.\n\n")
        
        # Write detailed statistics table
        f.write("-"*80 + "\n")
        f.write("Detailed Statistics:\n")
        f.write("-"*80 + "\n\n")
        
        # Create table header
        f.write(f"{'Statistic':<25} {'Random':>15} {'Induced':>15} {'Combined':>15}\n")
        f.write("-"*72 + "\n")
        
        # Add rows for each statistic
        stats_to_include = [
            ('mean', 'Mean Survivors'),
            ('variance', 'Variance'),
            ('std_dev', 'Standard Deviation'),
            ('median', 'Median'),
            ('max', 'Maximum'),
            ('coefficient_of_variation', 'Coef. of Variation'),
            ('variance_to_mean_ratio', 'VMR'),
            ('zero_resistant_cultures', 'Zero Resistant Count'),
            ('zero_resistant_percent', 'Zero Resistant %')
        ]
        
        for key, label in stats_to_include:
            # Handle VMR which might be calculated or directly stored
            if key == 'variance_to_mean_ratio':
                r_val = vmr_random
                i_val = vmr_induced
                c_val = vmr_combined
            else:
                r_val = stats['random'].get(key, 'N/A')
                i_val = stats['induced'].get(key, 'N/A')
                c_val = stats['combined'].get(key, 'N/A')
            
            # Format based on magnitude
            if isinstance(r_val, (int, float)) and r_val < 100:
                fmt = f"{label:<25} {r_val:>15.2f} {i_val:>15.2f} {c_val:>15.2f}\n"
            else:
                fmt = f"{label:<25} {r_val:>15} {i_val:>15} {c_val:>15}\n"
                
            f.write(fmt)
        
        f.write("\n")
        
        # Add execution time information if provided
        if elapsed_times:
            f.write("-"*80 + "\n")
            f.write("Execution Times:\n")
            f.write("-"*80 + "\n\n")
            
            f.write(f"{'Model':<15} {'Simulation':>15} {'Analysis':>15} {'Total':>15}\n")
            f.write("-"*62 + "\n")
            
            for model in ['random', 'induced', 'combined']:
                sim_time = elapsed_times.get(model, 0)
                analysis_time = elapsed_times.get(f"{model}_analysis", 0)
                total = sim_time + analysis_time
                
                f.write(f"{model.capitalize():<15} {sim_time:>15.2f}s {analysis_time:>15.2f}s {total:>15.2f}s\n")
            
            # Total time
            total_time = sum(elapsed_times.values())
            f.write(f"\nTotal execution time for all operations: {total_time:.2f} seconds\n\n")
        
        # Write interpretation
        f.write("="*80 + "\n")
        f.write("Interpretation:\n")
        f.write("="*80 + "\n\n")
        
        f.write("The key insight from Luria and Delbrück's experiment was that the distribution of mutants\n")
        f.write("in the random (Darwinian) model shows much greater variance than would be expected under\n")
        f.write("the induced (Lamarckian) model.\n\n")
        
        f.write("In the random model, mutations can occur at any generation during growth. Early mutations\n")
        f.write("lead to many resistant cells ('jackpots'), while late mutations produce few resistant cells.\n")
        f.write("This creates a highly skewed distribution with high variance.\n\n")
        
        f.write("In contrast, the induced model predicts that mutations only occur in response to selection\n")
        f.write("pressure, which would produce a Poisson distribution with lower variance (variance ≈ mean).\n\n")
        
        if vmr_random > vmr_induced:
            f.write("The simulation results support the historical findings: the random model shows significantly\n")
            f.write("higher variation, as measured by the variance-to-mean ratio (VMR).\n\n")
        else:
            f.write("NOTE: The simulation parameters may need adjustment, as the historical finding of higher\n")
            f.write("variance in the random model is not clearly demonstrated with the current settings.\n\n")
        
        f.write("The combined model demonstrates how both mechanisms could potentially operate together.\n\n")
        
        f.write("="*80 + "\n")
    
    log.info(f"Comparison report created at {report_path}")
    return report_path

def create_time_summary(elapsed_times, results_path):
    """Create a summary of execution times for different operations"""
    summary_path = os.path.join(results_path, "execution_time_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("EXECUTION TIME SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        # Group times by operation type
        simulation_times = {}
        analysis_times = {}
        visualization_times = {}
        other_times = {}
        
        for key, value in elapsed_times.items():
            if key in ['random', 'induced', 'combined']:
                simulation_times[key] = value
            elif '_analysis' in key:
                analysis_times[key] = value
            elif '_visualization' in key or 'comparison' in key:
                visualization_times[key] = value
            else:
                other_times[key] = value
        
        # Write simulation times
        f.write("Simulation Times:\n")
        f.write("-"*50 + "\n")
        for key, value in simulation_times.items():
            f.write(f"{key.capitalize()} model: {value:.2f} seconds\n")
        f.write("\n")
        
        # Write analysis times
        f.write("Analysis Times:\n")
        f.write("-"*50 + "\n")
        for key, value in analysis_times.items():
            model = key.replace('_analysis', '')
            f.write(f"{model.capitalize()} model analysis: {value:.2f} seconds\n")
        f.write("\n")
        
        # Write visualization times
        f.write("Visualization Times:\n")
        f.write("-"*50 + "\n")
        for key, value in visualization_times.items():
            f.write(f"{key}: {value:.2f} seconds\n")
        f.write("\n")
        
        # Write other times
        if other_times:
            f.write("Other Operations:\n")
            f.write("-"*50 + "\n")
            for key, value in other_times.items():
                f.write(f"{key}: {value:.2f} seconds\n")
            f.write("\n")
        
        # Write total time
        total_time = sum(elapsed_times.values())
        f.write("="*50 + "\n")
        f.write(f"Total execution time: {total_time:.2f} seconds\n")
    
    log.info(f"Execution time summary created at {summary_path}")
    return summary_path

#######################################################################
# Profiling and Optimization Functions
#######################################################################
def run_profiling(function_to_profile, *args, **kwargs):
    """Run a function with profiling to identify bottlenecks"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = function_to_profile(*args, **kwargs)
    
    profiler.disable()
    return result, profiler

def save_profiling_results(profiler, results_path, name="profile_results"):
    """Save profiling results to a file"""
    stats_path = os.path.join(results_path, f"{name}.prof")
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.dump_stats(stats_path)
    
    # Also create a text file with the top 20 time-consuming functions
    text_path = os.path.join(results_path, f"{name}.txt")
    with open(text_path, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats(20)
    
    log.info(f"Profiling results saved to {stats_path} and {text_path}")
    return stats_path, text_path

#######################################################################
# Main
#######################################################################
def main():
    parser = argparse.ArgumentParser(description="Luria–Delbrück Simulation: Random vs. Induced vs. Combined")
    parser.add_argument("--model", choices=["random", "induced", "combined", "all"], default="random", help="Which model to simulate? Use 'all' to run all models and compare")
    parser.add_argument("--num_cultures", type=int, default=1000, help="Number of replicate cultures")
    parser.add_argument("--initial_population_size", type=int, default=10, help="Initial population size per culture")
    parser.add_argument("--num_generations", type=int, default=5, help="Number of generations (cell divisions)")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate (used as either random, induced, or both)")
    parser.add_argument("--induced_mutation_rate", type=float, default=0.1, help="Separate induced mutation rate for the combined model")
    parser.add_argument("--n_genes", type=int, default=1, help="Number of genes in each organism's genome (first gene determines resistance)")
    parser.add_argument("--Results_path", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Path to the Results directory")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Logging level")
    parser.add_argument("--use_parallel", action="store_true", help="Use parallel processing for simulations")
    parser.add_argument("--n_processes", type=int, default=None, help="Number of processes to use (default: CPU count - 1)")
    parser.add_argument("--profile", action="store_true", help="Run with profiling to identify bottlenecks")
    parser.add_argument("--optimize_numpy", action="store_true", help="Use numpy vectorization where possible")
    parser.add_argument("--optimal_ld", action="store_true", help="Use optimal parameters for Luria-Delbrück demonstration")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # If optimal_ld flag is set, override parameters with values optimized for demonstrating Luria-Delbrück effect
    if args.optimal_ld:
        args.num_cultures = 5000
        args.initial_population_size = 100
        args.num_generations = 20
        args.mutation_rate = 0.0001
        args.induced_mutation_rate = 0.05
        print("Using optimal parameters for Luria-Delbrück demonstration:")
        print(f"  Number of cultures: {args.num_cultures}")
        print(f"  Initial population size: {args.initial_population_size}")
        print(f"  Number of generations: {args.num_generations}")
        print(f"  Random mutation rate: {args.mutation_rate}")
        print(f"  Induced mutation rate: {args.induced_mutation_rate}")

    # Record start time for overall execution
    start_time = time.time()

    # Set results path and create if needed
    if not os.path.exists(args.Results_path):
        os.makedirs(args.Results_path)
    
    # Create timestamped subdirectory with all parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir = os.path.join(
        args.Results_path,
        f"luria_delbruck_model_{args.model}_cultures_{args.num_cultures}_popsize_{args.initial_population_size}_gens_{args.num_generations}_mutrate_{args.mutation_rate}_indmutrate_{args.induced_mutation_rate}_ngenes_{args.n_genes}_parallel_{args.use_parallel}_{timestamp}"
    )
    os.makedirs(subdir, exist_ok=True)

    # Initialize logging with timestamp subfolder
    logger, args.results_path = init_log(subdir, args.log_level)

    # Apply _get_lsf_job_details to get the job details
    job_details = _get_lsf_job_details()
    if any(detail.split(': ')[1] for detail in job_details):
        # log the job details
        logger.info(f"Job details: {job_details}")
    else:
        logger.info("No LSF job details available")
    
    # Log simulation parameters
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting simulation on {date}")
    logger.info(f"Simulation parameters:")
    logger.info(f"Model: {args.model}")
    logger.info(f"Number of cultures: {args.num_cultures}")
    logger.info(f"Initial population size: {args.initial_population_size}")
    logger.info(f"Number of generations: {args.num_generations}")
    logger.info(f"Mutation rate: {args.mutation_rate}")
    logger.info(f"Parallel processing: {args.use_parallel}")
    if args.use_parallel:
        n_processes = args.n_processes or max(1, mp.cpu_count() - 1)
        logger.info(f"Number of processes: {n_processes}")
    if args.model == "combined" or args.model == "all":
        logger.info(f"Induced mutation rate: {args.induced_mutation_rate}")
    
    # Set up the Organism class parameters
    Organism.N_GENES = args.n_genes  
    Organism.P_MUTATION = args.mutation_rate
    
    # Run the appropriate simulation model
    if args.profile:
        logger.info("Running with profiling enabled")
        
        if args.model == "all":
            # Fix for pickling error: Don't profile the multi-model run
            logger.warning("Profiling not supported with multi-model run. Running without profiling.")
            results, stats, times = run_all_models(args)
        else:
            # Profile a single model run
            func = simulate_mutations
            kwargs = {
                "model_type": args.model,
                "num_cultures": args.num_cultures,
                "initial_population_size": args.initial_population_size,
                "num_generations": args.num_generations,
                "mutation_rate": args.mutation_rate,
                "use_parallel": args.use_parallel,
                "n_processes": args.n_processes
            }
            
            if args.model == "combined":
                kwargs["induced_mutation_rate"] = args.induced_mutation_rate
            
            # Run with profiling (handle the case if parallel processing is used)
            if args.use_parallel:
                # Can't profile with parallel processing due to pickling issues
                logger.warning("Cannot profile with parallel processing. Running without profiling.")
                result, _ = func(**kwargs)
                survivors_list = result
            else:
                # Run with profiling for sequential processing
                (survivors_list, _), profiler = run_profiling(func, **kwargs)
                save_profiling_results(profiler, args.results_path, f"{args.model}_profile")
            
            # Continue with analysis and visualization
            stats, _ = analyze_results(survivors_list, args.model)
            create_visualizations(survivors_list, args.model, args.results_path)
            
            # Log results
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{key}: {value:.4f}")
                else:
                    logger.info(f"{key}: {value}")
            
            # Print summary to console
            print(f"\nModel: {args.model}")
            print(f"Number of cultures: {args.num_cultures}")
            print(f"Mean # survivors: {stats['mean']:.4f}")
            print(f"Variance of # survivors: {stats['variance']:.4f}")
            print(f"Coefficient of variation: {stats['coefficient_of_variation']:.4f}")
            print(f"Cultures with zero resistant organisms: {stats['zero_resistant_percent']:.2f}%")
    
    elif args.model == "all":
        # Run all models and compare them
        results, stats, times = run_all_models(args)
        
        # Print summary to console
        print("\nSummary of Model Comparison:")
        print("-" * 40)
        for model in results.keys():
            model_stats = stats[model]
            vmr = model_stats['variance'] / model_stats['mean'] if model_stats['mean'] > 0 else 0
            
            print(f"{model.capitalize()} Model:")
            print(f"  Mean # survivors: {model_stats['mean']:.2f}")
            print(f"  Variance: {model_stats['variance']:.2f}")
            print(f"  Coefficient of Variation: {model_stats['coefficient_of_variation']:.2f}")
            print(f"  Variance-to-Mean Ratio: {vmr:.2f}")
            
            if model in times:
                print(f"  Simulation time: {times[model]:.2f} seconds")
            print("-" * 40)
        
        # Compare random vs induced VMR
        if 'random' in stats and 'induced' in stats:
            random_vmr = stats['random']['variance'] / stats['random']['mean'] if stats['random']['mean'] > 0 else 0
            induced_vmr = stats['induced']['variance'] / stats['induced']['mean'] if stats['induced']['mean'] > 0 else 0
            vmr_ratio = random_vmr / induced_vmr if induced_vmr > 0 else 0
            
            print(f"\nRandom/Induced VMR ratio: {vmr_ratio:.2f}")
            if random_vmr > induced_vmr:
                print("✓ Results support the Luria-Delbrück hypothesis")
            else:
                print("✗ Results do not clearly support the Luria-Delbrück hypothesis")
        
        # Indicate where results are saved
        print(f"\nDetailed results and visualizations saved to:")
        print(f"  {args.results_path}")
        
    else:
        # Run a single model
        logger.info(f"Running {args.model} mutation model simulation...")
        
        kwargs = {
            "model_type": args.model,
            "num_cultures": args.num_cultures,
            "initial_population_size": args.initial_population_size,
            "num_generations": args.num_generations,
            "mutation_rate": args.mutation_rate,
            "use_parallel": args.use_parallel,
            "n_processes": args.n_processes
        }
        
        if args.model == "combined":
            kwargs["induced_mutation_rate"] = args.induced_mutation_rate
            
        survivors_list, elapsed_time = simulate_mutations(**kwargs)
        
        # Log completion of simulation
        logger.info(f"Simulation completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Found {sum(1 for s in survivors_list if s > 0)} cultures having resistant organisms")

        # Run advanced analysis
        logger.info("Performing statistical analysis...")
        stats, stats_time = analyze_results(survivors_list, args.model)
        
        # Log the statistics
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        
        # Calculate variance-to-mean ratio (key for Luria-Delbrück analysis)
        vmr = stats['variance'] / stats['mean'] if stats['mean'] > 0 else 0
        logger.info(f"Variance-to-Mean Ratio: {vmr:.4f}")
        
        # Also print to console
        print(f"\nModel: {args.model}")
        print(f"Number of cultures: {args.num_cultures}")
        print(f"Mean # survivors: {stats['mean']:.4f}")
        print(f"Variance of # survivors: {stats['variance']:.4f}")
        print(f"Coefficient of variation: {stats['coefficient_of_variation']:.4f}")
        print(f"Variance-to-Mean Ratio: {vmr:.4f}")
        print(f"Cultures with zero resistant organisms: {stats['zero_resistant_percent']:.2f}%")
        print(f"Simulation time: {elapsed_time:.2f} seconds")
        print(f"Analysis time: {stats_time:.2f} seconds")
        
        # Create standard visualizations
        logger.info("Creating visualizations...")
        vis_path, vis_time = create_visualizations(survivors_list, args.model, args.results_path)
        if vis_path:
            print(f"Visualizations saved to: {vis_path}")
            print(f"Visualization time: {vis_time:.2f} seconds")
        
        # Create time summary
        elapsed_times = {
            args.model: elapsed_time,
            f"{args.model}_analysis": stats_time,
            f"{args.model}_visualization": vis_time
        }
        create_time_summary(elapsed_times, args.results_path)
        
        # Save raw data for future analysis
        try:
            data_path = os.path.join(args.results_path, f"survivors_{args.model}.csv")
            np.savetxt(data_path, survivors_list, fmt='%d', delimiter=',')
            logger.info(f"Raw survivor counts saved to {data_path}")
        except Exception as e:
            logger.error(f"Error saving raw data: {str(e)}")
        
    # Record and log total execution time
    end_time = time.time()
    total_execution_time = end_time - start_time
    logger.info(f"Total script execution time: {total_execution_time:.2f} seconds")
    print(f"\nTotal execution time: {total_execution_time:.2f} seconds")
    
    logger.info("Analysis complete.")
    print(f"\nAll output files saved to: {args.results_path}")
    
if __name__ == "__main__":
    main()