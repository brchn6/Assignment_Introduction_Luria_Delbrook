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
def process_culture(culture_id, initial_population_size, num_generations, mutation_rate, log_first_cultures=False):
    """Process a single culture for random mutations model - for parallel processing"""
    # Initialize population with sensitive organisms
    population = [Organism() for _ in range(initial_population_size)]
    
    # Track generation stats for this culture
    generation_stats = [] if culture_id == 0 and log_first_cultures else None
    
    # Simulate generations
    for gen in range(num_generations):
        new_population = []
        for organism in population:
            # Each organism produces two offspring
            offspring1 = organism.reproduce(mutation_rate)
            offspring2 = organism.reproduce(mutation_rate)
            
            new_population.append(offspring1)
            new_population.append(offspring2)
            
        population = new_population
        
        # Track statistics for this generation
        if generation_stats is not None:
            resistant_count = sum(org.is_resistant for org in population)
            resistant_percent = (resistant_count / len(population)) * 100
            generation_stats.append({
                'gen': gen + 1,
                'population': len(population),
                'resistant': resistant_count,
                'resistant_percent': resistant_percent
            })
    
    # Count survivors (resistant organisms) after all generations
    survivors = sum(org.is_resistant for org in population)
    
    # Return both survivors and generation stats if available
    return survivors, generation_stats if generation_stats else None

@timing_decorator
def simulate_random_mutations(num_cultures=1000, initial_population_size=10, num_generations=5, mutation_rate=0.1, use_parallel=True, n_processes=None):
    """
    Random (Darwinian) mutation model using the Organism class.
    Mutations arise spontaneously at each generation.
    Returns a list of 'survivors' (resistant cell counts) across cultures.
    """
    # Set the class variables for all Organisms
    Organism.P_MUTATION = mutation_rate
    
    survivors_list = []
    generation_stats = None
    log.debug(f"Starting random mutation simulation with {num_cultures} cultures")

    if use_parallel and num_cultures > 10:
        # Use multiprocessing for better performance
        n_processes = n_processes or max(1, mp.cpu_count() - 1)
        log.info(f"Using parallel processing with {n_processes} processes")
        
        # Create a partial function with fixed parameters
        process_func = partial(
            process_culture, 
            initial_population_size=initial_population_size,
            num_generations=num_generations,
            mutation_rate=mutation_rate,
            log_first_cultures=True
        )
        
        # Process cultures in parallel
        with mp.Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap(process_func, range(num_cultures)),
                total=num_cultures,
                desc="Random Model Progress",
                unit="culture"
            ))
        
        # Separate survivors and stats
        for survivors, stats in results:
            survivors_list.append(survivors)
            if stats is not None:
                generation_stats = stats
        
    else:
        # Sequential processing with progress bar
        for culture_id in tqdm(range(num_cultures), desc="Random Model Progress", unit="culture"):
            survivors, stats = process_culture(
                culture_id, 
                initial_population_size,
                num_generations,
                mutation_rate,
                log_first_cultures=True
            )
            survivors_list.append(survivors)
            if stats is not None:
                generation_stats = stats
    
    # Log generation progression for first culture
    if generation_stats:
        log.info("Generation progression (first culture):")
        for stat in generation_stats:
            log.info(f"  Generation {stat['gen']}: {stat['resistant']} resistant ({stat['resistant_percent']:.2f}%)")

    return survivors_list

def process_culture_induced(culture_id, initial_population_size, num_generations, mutation_rate, log_first_cultures=False):
    """Process a single culture for induced mutations model - for parallel processing"""
    # Initialize population with sensitive organisms
    population = [Organism() for _ in range(initial_population_size)]
    
    # Track generation stats for this culture
    generation_stats = [] if culture_id == 0 and log_first_cultures else None
    
    # Simulate generations with no mutations
    for gen in range(num_generations):
        new_population = []
        for organism in population:
            # Each organism reproduces without mutation
            offspring1 = Organism(genome=organism.genome.copy())
            offspring2 = Organism(genome=organism.genome.copy())
            
            new_population.append(offspring1)
            new_population.append(offspring2)
            
        population = new_population
        
        # Track statistics for this generation
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
    for organism in population:
        organism.mutate(mutation_rate)  # Apply with specified rate
    
    # Count survivors
    survivors = sum(org.is_resistant for org in population)
    
    # Return both survivors and generation stats if available
    return survivors, generation_stats if generation_stats else None

@timing_decorator
def simulate_induced_mutations(num_cultures=1000, initial_population_size=10, num_generations=5, mutation_rate=0.1, use_parallel=True, n_processes=None):
    """
    Induced (Lamarckian) mutation model using the Organism class.
    No mutations during growth; only at the end (when selective agent is applied).
    Returns a list of 'survivors' (resistant cell counts) across cultures.
    """
    # Set the class variables for all Organisms
    Organism.P_MUTATION = 0  # No spontaneous mutations during growth
    
    survivors_list = []
    generation_stats = None
    log.debug(f"Starting induced mutation simulation with {num_cultures} cultures")

    if use_parallel and num_cultures > 10:
        # Use multiprocessing for better performance
        n_processes = n_processes or max(1, mp.cpu_count() - 1)
        log.info(f"Using parallel processing with {n_processes} processes")
        
        # Create a partial function with fixed parameters
        process_func = partial(
            process_culture_induced, 
            initial_population_size=initial_population_size,
            num_generations=num_generations,
            mutation_rate=mutation_rate,
            log_first_cultures=True
        )
        
        # Process cultures in parallel
        with mp.Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap(process_func, range(num_cultures)),
                total=num_cultures,
                desc="Induced Model Progress",
                unit="culture"
            ))
        
        # Separate survivors and stats
        for survivors, stats in results:
            survivors_list.append(survivors)
            if stats is not None:
                generation_stats = stats
        
    else:
        # Sequential processing with progress bar
        for culture_id in tqdm(range(num_cultures), desc="Induced Model Progress", unit="culture"):
            survivors, stats = process_culture_induced(
                culture_id, 
                initial_population_size,
                num_generations,
                mutation_rate,
                log_first_cultures=True
            )
            survivors_list.append(survivors)
            if stats is not None:
                generation_stats = stats
    
    # Log generation progression for first culture
    if generation_stats:
        log.info("Generation progression (first culture):")
        for stat in generation_stats:
            log.info(f"  Generation {stat['gen']}: {stat['resistant']} resistant ({stat['resistant_percent']:.2f}%)")
        
        # Calculate final stats
        final_resistant = survivors_list[0]
        final_population = initial_population_size * (2 ** num_generations)
        final_percent = (final_resistant / final_population) * 100
        log.info(f"  After induced mutations: {final_resistant} resistant ({final_percent:.2f}%)")

    return survivors_list

def process_culture_combined(culture_id, initial_population_size, num_generations, random_mut_rate, induced_mut_rate, log_first_cultures=False):
    """Process a single culture for combined mutations model - for parallel processing"""
    # Initialize population with sensitive organisms
    population = [Organism() for _ in range(initial_population_size)]
    
    # Track generation stats for this culture
    generation_stats = [] if culture_id == 0 and log_first_cultures else None
    
    # Simulate generations with random mutations
    for gen in range(num_generations):
        new_population = []
        for organism in population:
            # Each organism produces two offspring with possible mutations
            offspring1 = organism.reproduce(random_mut_rate)
            offspring2 = organism.reproduce(random_mut_rate)
            
            new_population.append(offspring1)
            new_population.append(offspring2)
            
        population = new_population
        
        # Track statistics for this generation
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
        organism.mutate(induced_mut_rate)  # Apply with specified rate
    
    # Count survivors
    survivors = sum(org.is_resistant for org in population)
    
    # Return survivors, stats, and pre-induced count
    return survivors, generation_stats if generation_stats else None, pre_induced_resistant

@timing_decorator
def simulate_combined_mutations(num_cultures=1000, initial_population_size=10, num_generations=5, 
                               random_mut_rate=0.1, induced_mut_rate=0.1, use_parallel=True, n_processes=None):
    """
    Combined model using the Organism class:
    - Random mutations occur during each generation.
    - Additional induced mutations occur at the end.
    """
    # Set the class variables for all Organisms
    Organism.P_MUTATION = random_mut_rate
    
    survivors_list = []
    generation_stats = None
    pre_induced_resistant = 0
    log.debug(f"Starting combined mutation simulation with {num_cultures} cultures")

    if use_parallel and num_cultures > 10:
        # Use multiprocessing for better performance
        n_processes = n_processes or max(1, mp.cpu_count() - 1)
        log.info(f"Using parallel processing with {n_processes} processes")
        
        # Create a partial function with fixed parameters
        process_func = partial(
            process_culture_combined, 
            initial_population_size=initial_population_size,
            num_generations=num_generations,
            random_mut_rate=random_mut_rate,
            induced_mut_rate=induced_mut_rate,
            log_first_cultures=True
        )
        
        # Process cultures in parallel
        with mp.Pool(processes=n_processes) as pool:
            results = list(tqdm(
                pool.imap(process_func, range(num_cultures)),
                total=num_cultures,
                desc="Combined Model Progress",
                unit="culture"
            ))
        
        # Separate survivors, stats, and pre-induced count
        for i, (survivors, stats, pre_induced) in enumerate(results):
            survivors_list.append(survivors)
            if stats is not None:
                generation_stats = stats
            if i == 0:
                pre_induced_resistant = pre_induced
        
    else:
        # Sequential processing with progress bar
        for culture_id in tqdm(range(num_cultures), desc="Combined Model Progress", unit="culture"):
            survivors, stats, pre_induced = process_culture_combined(
                culture_id, 
                initial_population_size,
                num_generations,
                random_mut_rate,
                induced_mut_rate,
                log_first_cultures=True
            )
            survivors_list.append(survivors)
            if stats is not None:
                generation_stats = stats
            if culture_id == 0:
                pre_induced_resistant = pre_induced
    
    # Log generation progression and effect of induced mutations for first culture
    if generation_stats:
        post_induced_resistant = survivors_list[0]
        induced_effect = post_induced_resistant - pre_induced_resistant
        log.info("Generation progression (first culture):")
        for stat in generation_stats:
            log.info(f"  Generation {stat['gen']}: {stat['resistant']} resistant ({stat['resistant_percent']:.2f}%)")
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
        import numpy as np
        
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
            nonzero_data = [x + 0.1 for x in survivors_list]
            bins = np.logspace(np.log10(0.1), np.log10(max(nonzero_data)), 20)
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
                theoretical_data = np.random.lognormal(0, 1, size=1000)
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
    """Create clearer comparison visualizations for all models."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
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
    
    Organism.N_GENES = 1
    Organism.P_MUTATION = args.mutation_rate
    
    random_survivors, random_time = simulate_random_mutations(
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
    
    # Original visualizations
    _, vis_time = create_visualizations(random_survivors, 'random', args.results_path)
    elapsed_times['random_visualization'] = vis_time
    
    # New improved visualizations
    try:
        _, improved_vis_time = create_visualizations(random_survivors, 'random', args.results_path)
        elapsed_times['random_improved_visualization'] = improved_vis_time
    except Exception as e:
        log.error(f"Error creating improved visualizations for random model: {str(e)}")
    
    # 2. Run Induced Mutations Model
    log.info("\n" + "="*50)
    log.info("Running Induced Mutations Model")
    log.info("="*50)
    
    induced_survivors, induced_time = simulate_induced_mutations(
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
    
    # Original visualizations
    _, vis_time = create_visualizations(induced_survivors, 'induced', args.results_path)
    elapsed_times['induced_visualization'] = vis_time
    
    # New improved visualizations
    try:
        _, improved_vis_time = create_visualizations(induced_survivors, 'induced', args.results_path)
        elapsed_times['induced_improved_visualization'] = improved_vis_time
    except Exception as e:
        log.error(f"Error creating improved visualizations for induced model: {str(e)}")
    
    # 3. Run Combined Mutations Model
    log.info("\n" + "="*50)
    log.info("Running Combined Mutations Model")
    log.info("="*50)
    
    combined_survivors, combined_time = simulate_combined_mutations(
        num_cultures=args.num_cultures,
        initial_population_size=args.initial_population_size,
        num_generations=args.num_generations,
        random_mut_rate=args.mutation_rate,
        induced_mut_rate=args.induced_mutation_rate,
        use_parallel=args.use_parallel,
        n_processes=args.n_processes
    )
    
    results['combined'] = combined_survivors
    elapsed_times['combined'] = combined_time
    stats_result, stats_time = analyze_results(combined_survivors, 'combined')
    stats['combined'] = stats_result
    elapsed_times['combined_analysis'] = stats_time
    
    # Original visualizations
    _, vis_time = create_visualizations(combined_survivors, 'combined', args.results_path)
    elapsed_times['combined_visualization'] = vis_time
    
    # New improved visualizations
    try:
        _, improved_vis_time = create_visualizations(combined_survivors, 'combined', args.results_path)
        elapsed_times['combined_improved_visualization'] = improved_vis_time
    except Exception as e:
        log.error(f"Error creating improved visualizations for combined model: {str(e)}")
    
    # 4. Create original comparison visualizations
    log.info("\n" + "="*50)
    log.info("Creating Comparison Visualizations")
    log.info("="*50)
    
    _, comparison_time = create_comparison_visualization(results, args.results_path)
    elapsed_times['comparison_visualization'] = comparison_time
    
    # 5. Create new Luria-Delbrück direct comparison visualization
    try:
        log.info("Creating specialized Luria-Delbrück comparison visualization")
        _, ld_comparison_time = create_luria_delbrueck_comparison(results, args.results_path)
        elapsed_times['luria_delbrueck_comparison'] = ld_comparison_time
    except Exception as e:
        log.error(f"Error creating Luria-Delbrück comparison: {str(e)}")
    
    # 6. Calculate additional Luria-Delbrück specific metrics
    try:
        log.info("Calculating Luria-Delbrück specific metrics")
        ld_metrics_time_start = time.time()
        
        # Calculate variance to mean ratios for all models
        for model in results:
            data = results[model]
            mean = np.mean(data)
            variance = np.var(data)
            if mean > 0:
                vmr = variance / mean
                stats[model]['variance_to_mean_ratio'] = vmr
                log.info(f"{model.capitalize()} model variance-to-mean ratio: {vmr:.2f}")
                
                # For random model, this should be >> 1 for Luria-Delbrück distribution
                if model == 'random':
                    log.info(f"Random model VMR {vmr:.2f} {'> 1 as expected for Luria-Delbrück' if vmr > 1 else '< 1, unexpected'}")
                
                # For induced model, this should be ≈ 1 for Poisson distribution
                if model == 'induced':
                    log.info(f"Induced model VMR {vmr:.2f} {'≈ 1 as expected for Poisson' if 0.9 <= vmr <= 1.1 else 'not ≈ 1, unexpected'}")
        
        # Calculate p0 method mutation rate estimate for random model
        if 'random' in results:
            zero_count = sum(1 for x in results['random'] if x == 0)
            total_cultures = len(results['random'])
            if zero_count > 0 and zero_count < total_cultures:
                p0 = zero_count / total_cultures
                # Luria-Delbrück p0 method: mutation rate ≈ -ln(p0)/N where N is final population size
                final_pop_size = args.initial_population_size * (2 ** args.num_generations)
                mutation_rate_estimate = -np.log(p0) / final_pop_size
                log.info(f"Estimated mutation rate using p0 method: {mutation_rate_estimate:.8f}")
                stats['random']['p0_method_mutation_rate'] = mutation_rate_estimate
            else:
                log.info("Cannot estimate mutation rate using p0 method (no zero cultures or all zero cultures)")
        
        ld_metrics_time = time.time() - ld_metrics_time_start
        elapsed_times['luria_delbrueck_metrics'] = ld_metrics_time
    except Exception as e:
        log.error(f"Error calculating Luria-Delbrück metrics: {str(e)}")
    
    # 7. Create comparison report
    report_time_start = time.time()
    create_comparison_report(stats, args.results_path, elapsed_times)
    report_time = time.time() - report_time_start
    elapsed_times['report_creation'] = report_time
    
    # 8. Create enhanced report specifically for Luria-Delbrück analysis
    try:
        ld_report_time_start = time.time()
        create_luria_delbrueck_report(stats, results, args, args.results_path)
        ld_report_time = time.time() - ld_report_time_start
        elapsed_times['luria_delbrueck_report'] = ld_report_time
    except Exception as e:
        log.error(f"Error creating Luria-Delbrück report: {str(e)}")
    
    # 9. Create execution time summary
    create_time_summary(elapsed_times, args.results_path)
    
    log.info("\n" + "="*50)
    log.info("All models completed. Comparison analysis available at:")
    log.info(f"  {args.results_path}")
    log.info("="*50)
    
    total_time = sum(elapsed_times.values())
    log.info(f"Total execution time: {total_time:.2f} seconds")
    
    return results, stats, elapsed_times

def create_visualizations(survivors_list, model_name, results_path):
    """Create improved visualizations specifically for Luria-Delbrück analysis for individual models"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from matplotlib.gridspec import GridSpec
        
        # Set style for better visualizations
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(2, 2, figure=fig)
        
        # Color scheme based on model
        model_colors = {
            'random': 'royalblue', 
            'induced': 'firebrick', 
            'combined': 'forestgreen'
        }
        color = model_colors.get(model_name, 'purple')
        
        # Calculate key statistics for annotations
        mean_val = np.mean(survivors_list)
        var_val = np.var(survivors_list)
        std_val = np.std(survivors_list)
        cv_val = std_val / mean_val if mean_val > 0 else 0
        vmr_val = var_val / mean_val if mean_val > 0 else 0
        
        # 1. Standard histogram with KDE (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(survivors_list, kde=True, ax=ax1, color=color, alpha=0.7, edgecolor='k')
        ax1.set_xlabel("Number of Resistant Survivors")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"{model_name.capitalize()} Model: Distribution of Resistant Survivors", fontweight='bold')
        
        # Add statistics annotation
        stats_text = f"Mean: {mean_val:.2f}\nStd Dev: {std_val:.2f}\nCV: {cv_val:.2f}\nVMR: {vmr_val:.2f}"
        ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, 
                fontsize=11, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 2. Log-scale histogram - KEY for Luria-Delbrück (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        
        if max(survivors_list) > 0:
            # Handle zeros for log scale properly
            nonzero_data = [x for x in survivors_list if x > 0]
            if len(nonzero_data) > 0:
                # Add a small value for log binning to work
                nonzero_data = [x + 0.1 for x in nonzero_data]
                min_val = min(nonzero_data)
                max_val = max(nonzero_data)
                
                # Create log-spaced bins
                bins = np.logspace(np.log10(min_val), np.log10(max_val), 30)
                
                # Plot the histogram
                ax2.hist(nonzero_data, bins=bins, alpha=0.7, color=color, edgecolor='k')
                ax2.set_xscale('log')
                ax2.set_xlabel("Number of Resistant Survivors (log scale)")
                ax2.set_ylabel("Frequency")
                ax2.set_title("Log-Scale Distribution", fontweight='bold')
                
                # Add note explaining log scale
                ax2.text(0.02, 0.02, "Note: Log scale helps visualize\nskewed distributions", 
                        transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # For random model, add note about Luria-Delbrück signature
                if model_name == 'random' and vmr_val > 1:
                    ax2.text(0.98, 0.98, "Wide spread on log scale is\ncharacteristic of Luria-Delbrück\ndistribution", 
                            transform=ax2.transAxes, fontsize=10, ha='right', va='top',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        # 3. CCDF plot (bottom left) - Excellent for showing power-law type distributions
        ax3 = fig.add_subplot(gs[1, 0])
        sorted_data = np.sort(survivors_list)
        ccdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Plot with thicker line and markers for better visibility
        ax3.step(sorted_data, ccdf, where='post', color=color, linewidth=2.5)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel("Number of Resistant Survivors (log scale)")
        ax3.set_ylabel("P(X > x) (log scale)")
        ax3.set_title("Complementary Cumulative Distribution Function", fontweight='bold')
        ax3.grid(True, which="both", ls="-", alpha=0.2)
        
        # Add note on CCDF interpretation
        if model_name == 'random':
            note = "A straight line on log-log CCDF plot\nindicates a power-law-like distribution,\ncharacteristic of Luria-Delbrück"
        elif model_name == 'induced':
            note = "Curved line on log-log CCDF plot\nis characteristic of Poisson-like distributions\nexpected for induced mutations"
        else:
            note = "CCDF shape reveals distribution characteristics:\nStraight line = power-law-like (Luria-Delbrück)\nCurved line = Poisson-like"
            
        ax3.text(0.02, 0.02, note, transform=ax3.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 4. Theoretical comparison for specific models (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        
        if model_name in ["random", "induced"]:
            # Generate theoretical data for comparison
            if model_name == "random":
                # Use log-normal as an approximation to Luria-Delbrück distribution
                mean_log, sigma_log = 0, 1  # Parameters can be adjusted
                theoretical_data = np.random.lognormal(mean_log, sigma_log, size=10000)
                # Scale to match mean of empirical data
                scaling_factor = mean_val / np.mean(theoretical_data)
                theoretical_data = theoretical_data * scaling_factor
                
                ax4.set_title("Random Model vs. Theoretical Luria-Delbrück", fontweight='bold')
                note = "Luria-Delbrück distributions are characterized by:\n- High variance-to-mean ratio (VMR > 1)\n- Long right tail ('jackpot' events)\n- Similar to log-normal shape"
            else:
                # Simulate Poisson distribution for induced model
                theoretical_data = np.random.poisson(mean_val, size=10000)
                ax4.set_title("Induced Model vs. Theoretical Poisson", fontweight='bold')
                note = "Poisson distributions are characterized by:\n- VMR ≈ 1\n- More symmetrical distribution\n- More predictable outcomes"
            
            # Plot both distributions using KDE for smoother visualization
            sns.kdeplot(survivors_list, ax=ax4, color=color, linewidth=2.5, label="Simulation data")
            sns.kdeplot(theoretical_data, ax=ax4, color='gray', linewidth=2, linestyle='--', label="Theoretical distribution")
            
            # Set axis labels
            ax4.set_xlabel("Number of Resistant Survivors")
            ax4.set_ylabel("Probability Density")
            ax4.legend(loc='best')
            
            # Add note on theoretical comparison
            ax4.text(0.02, 0.02, note, transform=ax4.transAxes, fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Show VMR comparison
            theo_vmr = np.var(theoretical_data) / np.mean(theoretical_data)
            vmr_text = f"VMR Comparison:\n- Empirical: {vmr_val:.2f}\n- Theoretical: {theo_vmr:.2f}"
            ax4.text(0.98, 0.98, vmr_text, transform=ax4.transAxes, fontsize=10, 
                    ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        else:
            # For combined model, show histogram with density curve
            sns.histplot(survivors_list, kde=True, ax=ax4, color=color, alpha=0.6, edgecolor='k')
            ax4.set_xlabel("Number of Resistant Survivors")
            ax4.set_ylabel("Frequency")
            ax4.set_title("Combined Model Distribution", fontweight='bold')
            
            # Calculate quartiles for additional stats
            q1, median, q3 = np.percentile(survivors_list, [25, 50, 75])
            iqr = q3 - q1
            
            # Add detailed stats
            stats_text = (f"Distribution Statistics:\n"
                        f"- Mean: {mean_val:.2f}\n"
                        f"- Median: {median:.2f}\n"
                        f"- 1st Quartile: {q1:.2f}\n"
                        f"- 3rd Quartile: {q3:.2f}\n"
                        f"- IQR: {iqr:.2f}\n"
                        f"- VMR: {vmr_val:.2f}")
            ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes, fontsize=10, 
                    ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        save_path = os.path.join(results_path, f"improved_analysis_{model_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a specialized log-scale plot for better visualization of skewness
        plt.figure(figsize=(10, 8))
        
        # For random model, focus on log-log CCDF which best shows Luria-Delbrück characteristics
        if model_name == "random":
            plt.step(sorted_data, ccdf, where='post', color=color, linewidth=3)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("Number of Resistant Survivors (log scale)")
            plt.ylabel("P(X > x) (log scale)")
            plt.title("Random Model: Log-Log CCDF Plot", fontweight='bold')
            plt.grid(True, which="both", ls="-", alpha=0.3)
            
            # Add annotation explaining significance
            plt.figtext(0.5, 0.01, 
                        "A straight line on this log-log plot indicates a power-law-like distribution,\n"
                        "which is a key signature of the Luria-Delbrück experiment showing spontaneous mutations.",
                        ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        else:
            # For other models, show log-scale histogram
            nonzero_data = [x + 0.1 for x in survivors_list if x > 0]
            if len(nonzero_data) > 0:
                bins = np.logspace(np.log10(min(nonzero_data)), np.log10(max(nonzero_data)), 30)
                plt.hist(nonzero_data, bins=bins, alpha=0.7, color=color, edgecolor='k')
                plt.xscale('log')
                plt.xlabel("Number of Resistant Survivors (log scale)")
                plt.ylabel("Frequency")
                plt.title(f"{model_name.capitalize()} Model: Log-Scale Histogram", fontweight='bold')
                plt.grid(True, which="both", ls="-", alpha=0.3)
        
        plt.tight_layout()
        log_save_path = os.path.join(results_path, f"log_scale_analysis_{model_name}.png")
        plt.savefig(log_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path, 0  # Return path and time = 0 (timing handled by decorator)
        
    except ImportError as e:
        log.warning(f"Visualization libraries not available: {e}")
        return None, 0
    except Exception as e:
        log.error(f"Error creating improved visualizations: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        return None, 0
        
def create_luria_delbrueck_comparison(results_dict, results_path):
    """Create specialized visualizations comparing random vs induced models with improved clarity"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from matplotlib.gridspec import GridSpec
        
        if 'random' not in results_dict or 'induced' not in results_dict:
            return None, 0
            
        random_data = results_dict['random']
        induced_data = results_dict['induced']
        
        # Set the style for better aesthetics
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # Create figure with custom grid layout
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(2, 2, figure=fig)
        
        # 1. CCDF Comparison - most informative plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Process random data
        sorted_random = np.sort(random_data)
        ccdf_random = 1 - np.arange(1, len(sorted_random) + 1) / len(sorted_random)
        
        # Process induced data
        sorted_induced = np.sort(induced_data)
        ccdf_induced = 1 - np.arange(1, len(sorted_induced) + 1) / len(sorted_induced)
        
        # Plot both on same axes with thicker lines and markers for better visibility
        ax1.step(sorted_random, ccdf_random, where='post', color='blue', 
                label='Random (Darwinian) Model', linewidth=2.5)
        ax1.step(sorted_induced, ccdf_induced, where='post', color='red', 
                label='Induced (Lamarckian) Model', linewidth=2.5, linestyle='--')
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel("Number of Resistant Survivors (log scale)")
        ax1.set_ylabel("P(X > x) (log scale)")
        ax1.set_title("Complementary Cumulative Distribution Function", fontweight='bold')
        ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        
        # 2. Separate PDF plots for better visibility (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Create two separate subplots within this area
        divider = make_axes_locatable(ax2)
        ax2_random = ax2
        ax2_induced = divider.append_axes("bottom", size="40%", pad=0.6)
        
        # Plot Random model PDF
        sns.kdeplot(random_data, ax=ax2_random, color='blue', linewidth=2.5, label="Random Model")
        ax2_random.set_xlabel("")  # Remove xlabel as it will be on the bottom subplot
        ax2_random.set_ylabel("Probability Density")
        ax2_random.set_title("Probability Density Functions (Separate Scales)", fontweight='bold')
        ax2_random.legend(loc='upper right')
        ax2_random.grid(True, alpha=0.3)
        
        # Plot Induced model PDF
        sns.kdeplot(induced_data, ax=ax2_induced, color='red', linewidth=2.5, label="Induced Model")
        ax2_induced.set_xlabel("Number of Resistant Survivors")
        ax2_induced.set_ylabel("Probability Density")
        ax2_induced.legend(loc='upper right')
        ax2_induced.grid(True, alpha=0.3)
        
        # 3. Separate histograms with appropriate scales (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Create separate subplot areas
        divider = make_axes_locatable(ax3)
        ax3_random = ax3
        ax3_induced = divider.append_axes("right", size="30%", pad=0.6)
        
        # Plot Random model histogram
        ax3_random.hist(random_data, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax3_random.set_xlabel("Number of Resistant Survivors")
        ax3_random.set_ylabel("Frequency")
        ax3_random.set_title("Histograms (Separate Scales)", fontweight='bold')
        ax3_random.text(0.05, 0.95, "Random Model", transform=ax3_random.transAxes, 
                 fontsize=12, fontweight='bold', verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Plot Induced model histogram
        ax3_induced.hist(induced_data, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax3_induced.set_xlabel("Number")
        ax3_induced.set_ylabel("")  # No ylabel on right plot
        ax3_induced.text(0.05, 0.95, "Induced Model", transform=ax3_induced.transAxes, 
                 fontsize=12, fontweight='bold', verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.5))
        
        # 4. Key statistics in a styled text box (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        # Calculate statistics for both distributions
        random_mean = np.mean(random_data)
        random_var = np.var(random_data)
        random_std = np.std(random_data)
        random_cv = random_std / random_mean if random_mean > 0 else 0
        random_vmr = random_var / random_mean if random_mean > 0 else 0
        
        induced_mean = np.mean(induced_data)
        induced_var = np.var(induced_data)
        induced_std = np.std(induced_data)
        induced_cv = induced_std / induced_mean if induced_mean > 0 else 0
        induced_vmr = induced_var / induced_mean if induced_mean > 0 else 0
        
        # Create VMR comparison bar chart
        divider = make_axes_locatable(ax4)
        ax4_chart = divider.append_axes("bottom", size="40%", pad=0.2)
        
        models = ['Random', 'Induced']
        vmr_values = [random_vmr, induced_vmr]
        colors = ['blue', 'red']
        
        bars = ax4_chart.bar(models, vmr_values, color=colors, alpha=0.7, edgecolor='black')
        ax4_chart.set_yscale('log')  # Log scale for better comparison with high disparity
        ax4_chart.set_ylabel('VMR (log scale)')
        ax4_chart.set_title('Variance-to-Mean Ratio Comparison', fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4_chart.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                   f'{height:.2f}', ha='center', va='bottom', rotation=0,
                   fontweight='bold')
        
        # Create a text summary
        text = f
        """
        LURIA-DELBRUECK ANALYSIS RESULTS:
        
        RANDOM (DARWINIAN) MODEL:
        * Mean: {random_mean:.2f}
        * Standard Deviation: {random_std:.2f} 
        * Variance: {random_var:.2f}
        * Coefficient of Variation: {random_cv:.2f}
        * Variance-to-Mean Ratio: {random_vmr:.2f}
        * Zero resistant cultures: {sum(1 for x in random_data if x == 0)} 
        
        INDUCED (LAMARCKIAN) MODEL:
        * Mean: {induced_mean:.2f}
        * Standard Deviation: {induced_std:.2f}
        * Variance: {induced_var:.2f}
        * Coefficient of Variation: {induced_cv:.2f}
        * Variance-to-Mean Ratio: {induced_vmr:.2f}
        * Zero resistant cultures: {sum(1 for x in induced_data if x == 0)}
        
        KEY FINDING: {'Random model has higher variance (supporting Luria-Delbrueck)' 
                    if random_vmr > induced_vmr else 'Results inconclusive'}
        
        RATIO COMPARISON:
        * Random/Induced VMR Ratio: {random_vmr/induced_vmr:.2f}
        * Random/Induced CV Ratio: {random_cv/induced_cv:.2f}
        """
        
        # Add styled text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax4.text(0.05, 0.95, text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', 
                bbox=props, linespacing=1.5)
        
        plt.tight_layout()
        save_path = os.path.join(results_path, "luria_delbrueck_direct_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a second figure for log-scale comparisons
        fig2 = plt.figure(figsize=(12, 10))
        
        # Log-scale histogram comparison
        ax1 = fig2.add_subplot(111)
        
        # Handle zeros for log scale
        nonzero_random = [x + 0.1 for x in random_data if x > 0]
        nonzero_induced = [x + 0.1 for x in induced_data if x > 0]
        
        max_value = max(max(nonzero_random), max(nonzero_induced))
        min_value = min(min(nonzero_random), min(nonzero_induced))
        
        bins = np.logspace(np.log10(min_value), np.log10(max_value), 30)
        
        ax1.hist(nonzero_random, bins=bins, alpha=0.5, color='blue', label="Random Model")
        ax1.hist(nonzero_induced, bins=bins, alpha=0.5, color='red', label="Induced Model")
        ax1.set_xscale('log')
        ax1.set_xlabel("Number of Resistant Survivors (log scale)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Log-Scale Distribution Comparison", fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, which="both", alpha=0.3)
        
        # Add annotation explaining the log scale
        explanation = """
                    Note: This plot uses logarithmic scaling on the x-axis to better
                    display the distributions when they have very different ranges.
                    The Random (Darwinian) model typically shows a wider spread.
                    """
        
        ax1.text(0.02, 0.02, explanation, transform=ax1.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        log_save_path = os.path.join(results_path, "luria_delbrueck_log_comparison.png")
        plt.savefig(log_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path, 0
    except ImportError as e:
        log.warning(f"Visualization libraries not available: {e}")
        return None, 0
    except Exception as e:
        log.error(f"Error creating comparison visualizations: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        return None, 0

def create_luria_delbrueck_report(stats, results, args, results_path):
    """Create a report specifically focused on Luria-Delbrück analysis"""
    report_path = os.path.join(results_path, "luria_delbrueck_detailed_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LURIA-DELBRÜCK EXPERIMENT: DETAILED ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # Write introduction
        f.write("BACKGROUND\n")
        f.write("-"*80 + "\n\n")
        f.write("The Luria-Delbrück experiment (1943) was a pivotal study that demonstrated\n")
        f.write("that mutations in bacteria occur spontaneously rather than as a direct response\n") 
        f.write("to selection. The key insight was that spontaneous mutations occurring during\n")
        f.write("bacterial growth would produce a characteristic distribution of mutant counts\n")
        f.write("with high variance ('jackpot' cultures), while induced mutations would produce\n")
        f.write("a more uniform distribution (approximately Poisson).\n\n")
        
        # Write simulation parameters
        f.write("SIMULATION PARAMETERS\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Number of cultures: {args.num_cultures}\n")
        f.write(f"Initial population size: {args.initial_population_size}\n")
        f.write(f"Number of generations: {args.num_generations}\n")
        f.write(f"Random mutation rate: {args.mutation_rate}\n")
        f.write(f"Induced mutation rate: {args.induced_mutation_rate}\n")
        final_pop_size = args.initial_population_size * (2 ** args.num_generations)
        f.write(f"Final population size per culture: {final_pop_size}\n\n")
        
        # Key results focusing on Luria-Delbrück analysis
        f.write("KEY RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        # Calculate variance-to-mean ratios
        random_mean = stats['random']['mean']
        random_var = stats['random']['variance']
        random_vmr = random_var / random_mean if random_mean > 0 else 0
        
        induced_mean = stats['induced']['mean']
        induced_var = stats['induced']['variance']
        induced_vmr = induced_var / induced_mean if induced_mean > 0 else 0
        
        # Determine which distribution patterns are observed
        f.write("Distribution Characteristics:\n\n")
        
        # Random model analysis
        f.write("1. Random (Darwinian) Model:\n")
        f.write(f"   - Mean mutants per culture: {random_mean:.2f}\n")
        f.write(f"   - Variance: {random_var:.2f}\n")
        f.write(f"   - Variance-to-Mean Ratio: {random_vmr:.2f}\n")
        
        if random_vmr > 1.5:
            f.write("   - Result: High variance-to-mean ratio SUPPORTS the Luria-Delbrück hypothesis\n")
            f.write("     This indicates spontaneous mutations during growth (Darwinian model)\n")
        elif random_vmr > 1.0:
            f.write("   - Result: Moderate variance-to-mean ratio weakly supports the Luria-Delbrück hypothesis\n")
            f.write("     Consider increasing generations or adjusting mutation rate for clearer results\n")
        else:
            f.write("   - Result: Low variance-to-mean ratio does NOT support the Luria-Delbrück hypothesis\n")
            f.write("     Consider reviewing simulation parameters\n")
        
        # Check for jackpot cultures
        if 'random' in results:
            random_data = results['random']
            max_random = max(random_data)
            mean_random = np.mean(random_data)
            
            if max_random > 5 * mean_random:
                f.write(f"   - 'Jackpot' cultures detected: Max value ({max_random}) > 5× mean ({mean_random:.2f})\n")
                f.write("     This is characteristic of the Luria-Delbrück distribution\n")
            else:
                f.write(f"   - No strong 'jackpot' cultures detected: Max value ({max_random}) not much larger than mean\n")
                f.write("     Consider increasing generations or population size\n")
        f.write("\n")
        
        # Induced model analysis
        f.write("2. Induced (Lamarckian) Model:\n")
        f.write(f"   - Mean mutants per culture: {induced_mean:.2f}\n")
        f.write(f"   - Variance: {induced_var:.2f}\n")
        f.write(f"   - Variance-to-Mean Ratio: {induced_vmr:.2f}\n")
        
        if 0.9 <= induced_vmr <= 1.1:
            f.write("   - Result: Variance-to-Mean ratio ≈ 1 indicates a Poisson-like distribution\n")
            f.write("     This is expected for the induced mutation model\n")
        else:
            f.write(f"   - Result: Variance-to-Mean ratio = {induced_vmr:.2f} deviates from the expected Poisson-like distribution\n")
            f.write("     This may indicate simulation parameters need adjustment\n")
        f.write("\n")
        
        # Comparative analysis
        f.write("3. Comparative Analysis:\n")
        if random_vmr > induced_vmr * 2:
            f.write(f"   - Random model VMR ({random_vmr:.2f}) is > 2× Induced model VMR ({induced_vmr:.2f})\n")
            f.write("   - Result: STRONG SUPPORT for the Luria-Delbrück hypothesis\n")
        elif random_vmr > induced_vmr:
            f.write(f"   - Random model VMR ({random_vmr:.2f}) > Induced model VMR ({induced_vmr:.2f})\n")
            f.write("   - Result: MODERATE SUPPORT for the Luria-Delbrück hypothesis\n")
        else:
            f.write(f"   - Random model VMR ({random_vmr:.2f}) <= Induced model VMR ({induced_vmr:.2f})\n")
            f.write("   - Result: DOES NOT SUPPORT the Luria-Delbrück hypothesis\n")
            f.write("     Consider adjusting simulation parameters\n")
        f.write("\n")
        
        # Mutation rate estimation
        f.write("MUTATION RATE ESTIMATION\n")
        f.write("-"*80 + "\n\n")
        
        # p0 method
        f.write("1. p0 Method (Luria-Delbrück):\n")
        if 'random' in results:
            zero_count = sum(1 for x in results['random'] if x == 0)
            total_cultures = len(results['random'])
            if zero_count > 0 and zero_count < total_cultures:
                p0 = zero_count / total_cultures
                mutation_rate_estimate = -np.log(p0) / final_pop_size
                f.write(f"   - Cultures with zero mutants: {zero_count} ({p0*100:.2f}% of total)\n")
                f.write(f"   - Estimated mutation rate: {mutation_rate_estimate:.8f}\n")
                f.write(f"   - Actual mutation rate used: {args.mutation_rate:.8f}\n")
                accuracy = (mutation_rate_estimate / args.mutation_rate) * 100
                f.write(f"   - Estimation accuracy: {accuracy:.2f}%\n")
            else:
                f.write("   - Cannot calculate: All cultures have mutants or all cultures have zero mutants\n")
        f.write("\n")
        
        # Use median method
        if 'random' in results:
            random_data = results['random']
            median_value = np.median(random_data)
            if median_value > 0:
                # Using median approximation formula (simplified)
                median_estimate = np.log(2) / (final_pop_size * median_value)
                f.write("2. Median Method:\n")
                f.write(f"   - Median mutants per culture: {median_value}\n")
                f.write(f"   - Estimated mutation rate: {median_estimate:.8f}\n")
                f.write(f"   - Actual mutation rate used: {args.mutation_rate:.8f}\n")
                accuracy = (median_estimate / args.mutation_rate) * 100
                f.write(f"   - Estimation accuracy: {accuracy:.2f}%\n\n")
        
        # Write visualization guide
        f.write("VISUALIZATION GUIDE\n")
        f.write("-"*80 + "\n\n")
        f.write("The following visualizations are particularly informative for Luria-Delbrück analysis:\n\n")
        f.write("1. Log-Log CCDF Plot:\n")
        f.write("   - A straight line on this plot indicates a power-law-like distribution\n")
        f.write("   - Random mutation model should show a more linear pattern\n")
        f.write("   - Induced mutation model should show more curvature (Poisson-like)\n\n")
        
        f.write("2. Log-Scale Histogram:\n")
        f.write("   - Look for the 'long tail' in the random mutation model\n")
        f.write("   - This represents 'jackpot' cultures with many more mutants than average\n\n")
        
        f.write("3. Theoretical Distribution Comparison:\n")
        f.write("   - Random model is compared to a log-normal approximation (similar shape to Luria-Delbrück)\n")
        f.write("   - Induced model is compared to a Poisson distribution\n\n")
        
        # Write conclusion
        f.write("CONCLUSION\n")
        f.write("-"*80 + "\n\n")
        if random_vmr > induced_vmr:
            f.write("The simulation results SUPPORT the Luria-Delbrück hypothesis that mutations occur\n")
            f.write("randomly during growth rather than in response to selection. This is evidenced by:\n")
            f.write(f"1. Higher variance-to-mean ratio in the random model ({random_vmr:.2f}) compared\n")
            f.write(f"   to the induced model ({induced_vmr:.2f})\n")
            if 'random' in results and max(results['random']) > 5 * np.mean(results['random']):
                f.write("2. The presence of 'jackpot' cultures in the random model\n")
            f.write("3. Distribution patterns characteristic of spontaneous mutation\n\n")
        else:
            f.write("The current simulation parameters do not clearly demonstrate the Luria-Delbrück effect.\n")
            f.write("To better observe the characteristic distribution patterns, consider:\n")
            f.write("1. Reducing the mutation rate (e.g., to 0.0001)\n")
            f.write("2. Increasing the number of generations (e.g., to 15-20)\n")
            f.write("3. Increasing the initial population size (e.g., to 50-100)\n")
            f.write("4. Increasing the number of cultures (e.g., to 5000)\n\n")
        
        f.write("HISTORICAL CONTEXT\n")
        f.write("-"*80 + "\n\n")
        f.write("The Luria-Delbrück experiment provided compelling evidence against the Lamarckian view\n")
        f.write("that bacteria develop resistance in response to antibiotics or phages. Instead, it\n")
        f.write("supported the Darwinian view that random mutations conferring resistance occur\n")
        f.write("spontaneously during normal growth, before exposure to selective agents.\n\n")
        
        f.write("This experiment, known as the 'fluctuation test,' was a cornerstone in establishing\n")
        f.write("that bacteria follow the same evolutionary principles as higher organisms. Luria and\n")
        f.write("Delbrück were awarded the Nobel Prize in 1969 for this and other pioneering work in\n")
        f.write("bacterial genetics.\n\n")
        
        f.write("The mathematical framework they developed continues to be used today to estimate\n")
        f.write("mutation rates in bacteria and other microorganisms.\n")
    
    log.info(f"Luria-Delbrück detailed report created at {report_path}")
    return report_path

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
        
        # Determine key findings
        if cv_random > cv_induced:
            f.write(f"- The Random model shows higher variance (CV = {cv_random:.2f}) than the Induced model ")
            f.write(f"(CV = {cv_induced:.2f}), consistent with Luria & Delbrück's findings supporting Darwinian evolution.\n")
        else:
            f.write(f"- NOTE: Unexpected result - the Induced model shows higher variance than Random model.\n")
            f.write(f"  This may be due to parameter choices or statistical fluctuation in the simulation.\n")
        
        f.write(f"- The Combined model (CV = {cv_combined:.2f}) exhibits characteristics of both mechanisms.\n\n")
        
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
            ('zero_resistant_cultures', 'Zero Resistant Count'),
            ('zero_resistant_percent', 'Zero Resistant %')
        ]
        
        for key, label in stats_to_include:
            r_val = stats['random'][key]
            i_val = stats['induced'][key]
            c_val = stats['combined'][key]
            
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
        
        if cv_random > cv_induced:
            f.write("The simulation results support the historical findings: the random model shows significantly\n")
            f.write("higher variation, as measured by the coefficient of variation (CV = std.dev/mean).\n\n")
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
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir = os.path.join(
        args.Results_path,
        f"params_model_{args.model}_num_cultures_{args.num_cultures}_pop_size_{args.initial_population_size}_gens_{args.num_generations}_mut_rate_{args.mutation_rate}_induced_mut_rate_{args.induced_mutation_rate}_n_genes_{args.n_genes}_parallel_{args.use_parallel}_timestamp_{timestamp}"
    )
    os.makedirs(subdir, exist_ok=True)

    # Initialize logging with timestamp subfolder
    logger, args.results_path = init_log(subdir, args.log_level)

    # apply _get_lsf_job_details to get the job details
    job_details = _get_lsf_job_details()
    if job_details:
        # log the job details
        logger.info(f"Job details: {job_details}")
    else:
        logger.info("Job details not found")
    
    # Log simulation parameters
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_time = datetime.now().strftime("%H:%M:%S")
    logger.info(f"Starting simulation on this date {date} the time is {current_time}")
    logger.info(f"The name of the file is {__file__}")
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
            if args.model == "random":
                func = simulate_random_mutations
                kwargs = {
                    "num_cultures": args.num_cultures,
                    "initial_population_size": args.initial_population_size,
                    "num_generations": args.num_generations,
                    "mutation_rate": args.mutation_rate,
                    "use_parallel": args.use_parallel,
                    "n_processes": args.n_processes
                }
            elif args.model == "induced":
                func = simulate_induced_mutations
                kwargs = {
                    "num_cultures": args.num_cultures,
                    "initial_population_size": args.initial_population_size,
                    "num_generations": args.num_generations,
                    "mutation_rate": args.mutation_rate,
                    "use_parallel": args.use_parallel,
                    "n_processes": args.n_processes
                }
            else:  # "combined"
                func = simulate_combined_mutations
                kwargs = {
                    "num_cultures": args.num_cultures,
                    "initial_population_size": args.initial_population_size,
                    "num_generations": args.num_generations,
                    "random_mut_rate": args.mutation_rate,
                    "induced_mut_rate": args.induced_mutation_rate,
                    "use_parallel": args.use_parallel,
                    "n_processes": args.n_processes
                }
            
            # Run with profiling (handle the case if parallel processing is used)
            if args.use_parallel:
                # Can't profile with parallel processing due to pickling issues
                logger.warning("Cannot profile with parallel processing. Running without profiling.")
                result = func(**kwargs)
                survivors_list, _ = result
            else:
                # Run with profiling for sequential processing
                (survivors_list, _), profiler = run_profiling(func, **kwargs)
                save_profiling_results(profiler, args.results_path, f"{args.model}_profile")
            
            # Continue with analysis and visualization
            stats = analyze_results(survivors_list, args.model)[0]
            create_visualizations(survivors_list, args.model, args.results_path)
            
            # Try to use the new visualization functions if they exist
            try:
                create_visualizations(survivors_list, args.model, args.results_path)
            except Exception as e:
                logger.warning(f"Could not create improved visualizations: {str(e)}")
            
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
        # Fix for the pickling error in multiprocessing
        if args.use_parallel:
            # Set a flag to indicate functions should not use decorators
            os.environ["NO_TIMING_DECORATOR"] = "1"
            
        # Run all models and compare them
        result_tuple, elapsed_time = run_all_models(args)
        results, stats, times = result_tuple
        
        # Print summary to console
        print("\nSummary of Model Comparison:")
        print("-" * 40)
        for model in results.keys():
            model_stats = stats[model]
            print(f"{model.capitalize()} Model:")
            print(f"  Mean # survivors: {model_stats['mean']:.2f}")
            print(f"  Variance: {model_stats['variance']:.2f}")
            print(f"  Coefficient of Variation: {model_stats['coefficient_of_variation']:.2f}")
            
            # Calculate and display variance-to-mean ratio for Luria-Delbrück analysis
            vmr = model_stats['variance'] / model_stats['mean'] if model_stats['mean'] > 0 else 0
            print(f"  Variance-to-Mean Ratio: {vmr:.2f}")
            
            if 'model' in times:
                print(f"  Simulation time: {times[model]:.2f} seconds")
            print("-" * 40)
        
        # Indicate where results are saved
        print(f"\nDetailed results and visualizations saved to:")
        print(f"  {args.results_path}")
        
        # Reset the environment variable
        if "NO_TIMING_DECORATOR" in os.environ:
            del os.environ["NO_TIMING_DECORATOR"]
        
    else:
        # Run a single model
        logger.info(f"Running {args.model} mutation model simulation...")
        
        if args.model == "random":
            survivors_list, elapsed_time = simulate_random_mutations(
                num_cultures=args.num_cultures,
                initial_population_size=args.initial_population_size,
                num_generations=args.num_generations,
                mutation_rate=args.mutation_rate,
                use_parallel=args.use_parallel,
                n_processes=args.n_processes
            )
        elif args.model == "induced":
            survivors_list, elapsed_time = simulate_induced_mutations(
                num_cultures=args.num_cultures,
                initial_population_size=args.initial_population_size,
                num_generations=args.num_generations,
                mutation_rate=args.mutation_rate,
                use_parallel=args.use_parallel,
                n_processes=args.n_processes
            )
        else:  # "combined"
            survivors_list, elapsed_time = simulate_combined_mutations(
                num_cultures=args.num_cultures,
                initial_population_size=args.initial_population_size,
                num_generations=args.num_generations,
                random_mut_rate=args.mutation_rate,
                induced_mut_rate=args.induced_mutation_rate,
                use_parallel=args.use_parallel,
                n_processes=args.n_processes
            )
        
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
            print(f"Standard visualizations saved to: {vis_path}")
            print(f"Visualization time: {vis_time:.2f} seconds")
        
        # Try to use the new visualization functions if they exist
        try:
            improved_vis_path, improved_vis_time = create_visualizations(survivors_list, args.model, args.results_path)
            print(f"Improved Luria-Delbrück visualizations saved to: {improved_vis_path}")
        except Exception as e:
            logger.warning(f"Could not create improved visualizations: {str(e)}")
        
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
    
#######################################################################
# Entry point
#######################################################################
if __name__ == "__main__":
    main()


#######################################################################
# Example usage
#######################################################################
# python main.py --model random --num_cultures 1000 --initial_population_size 10 --num_generations 5 --mutation_rate 0.01 --Results_path results --log_level INFO --use_parallel
# python main.py --model induced --num_cultures 1000 --initial_population_size 10 --num_generations 5 --mutation_rate 0.01 --Results_path results --log_level INFO --use_parallel
# python main.py --model combined --num_cultures 1000 --initial_population_size 10 --num_generations 5 --mutation_rate 0.01 --induced_mutation_rate 0.01 --Results_path results --log_level INFO --use_parallel
# python main.py --model all --num_cultures 1000 --initial_population_size 10 --num_generations 5 --mutation_rate 0.01 --induced_mutation_rate 0.01 --Results_path results --log_level INFO --use_parallel
# python main.py --model all --num_cultures 1000 --initial_population_size 10 --num_generations 5 --mutation_rate 0.01 --induced_mutation_rate 0.01 --Results_path results --log_level INFO --profile