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
        
        # If the result is already a tuple, we need to handle it properly
        if isinstance(result, tuple):
            return result  # Return the original tuple without wrapping it
        else:
            return result, elapsed_time  # Wrap single result with elapsed time
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
# Enhanced Visualization Functions
#######################################################################
@timing_decorator
def create_visualizations(survivors_list, model_name, results_path):
    """Create enhanced visualizations of the simulation results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        
        # Set style for better visualizations
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 12))
        
        # 1. Standard histogram (top left)
        plt.subplot(2, 2, 1)
        plt.hist(survivors_list, bins='auto', alpha=0.7, color='royalblue')
        plt.xlabel("Number of Resistant Survivors", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"{model_name.capitalize()} Model: Distribution of Resistant Survivors", fontsize=14)
        
        # Calculate key statistics for annotations
        mean_val = np.mean(survivors_list)
        var_val = np.var(survivors_list)
        vmr_val = var_val / mean_val if mean_val > 0 else 0
        
        # Add statistics annotation
        stats_text = f"Mean: {mean_val:.2f}\nVariance: {var_val:.2f}\nVMR: {vmr_val:.2f}"
        plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 2. Log-scale histogram - KEY for Luria-Delbrück (top right)
        plt.subplot(2, 2, 2)
        if max(survivors_list) > 0:
            # Handle zeros for log scale
            nonzero_data = [x + 0.1 for x in survivors_list if x > 0]
            if len(nonzero_data) > 0:
                bins = np.logspace(np.log10(min(nonzero_data)), np.log10(max(nonzero_data)), 20)
                plt.hist(nonzero_data, bins=bins, alpha=0.7, color='forestgreen')
                plt.xscale('log')
                plt.xlabel("Number of Resistant Survivors (log scale)", fontsize=12)
                plt.ylabel("Frequency", fontsize=12)
                plt.title("Log-Scale Distribution", fontsize=14)
                
                # Add note explaining log scale significance
                if model_name == "random":
                    log_note = "Long-tailed distribution on log scale\nis characteristic of Luria-Delbrück effect"
                elif model_name == "induced":
                    log_note = "Narrow distribution on log scale\nis characteristic of Poisson distribution"
                else:
                    log_note = "Log scale helps visualize distribution shape"
                    
                plt.annotate(log_note, xy=(0.02, 0.02), xycoords='axes fraction',
                           fontsize=10, verticalalignment='bottom',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 3. CCDF plot (bottom left) - Excellent for showing power-law type distributions
        plt.subplot(2, 2, 3)
        sorted_data = np.sort(survivors_list)
        ccdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.step(sorted_data, ccdf, where='post', color='darkorange', linewidth=2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Number of Resistant Survivors (log scale)", fontsize=12)
        plt.ylabel("P(X > x) (log scale)", fontsize=12)
        plt.title("Complementary Cumulative Distribution Function", fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # Add annotation explaining CCDF significance
        if model_name == "random":
            ccdf_note = "Straight line on log-log CCDF plot\nindicates power-law behavior\ncharacteristic of Luria-Delbrück"
        elif model_name == "induced":
            ccdf_note = "Curved line on log-log CCDF plot\nindicates Poisson-like behavior"
        else:
            ccdf_note = "CCDF shape reveals underlying distribution pattern"
            
        plt.annotate(ccdf_note, xy=(0.02, 0.02), xycoords='axes fraction',
                   fontsize=10, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 4. Probability density function with theoretical fit (bottom right)
        plt.subplot(2, 2, 4)
        sns.kdeplot(survivors_list, color='purple', label="Empirical distribution", 
            linewidth=2, bw_adjust=2)  # Increase bw_adjust for more smoothing
        
        # Add theoretical distribution
        if model_name == "random":
            # Log-normal approximation for Luria-Delbrück
            nonzero_data = [x for x in survivors_list if x > 0]
            if len(nonzero_data) > 0:
                log_data = np.log(nonzero_data)
                mean_log = np.mean(log_data)
                sigma_log = np.std(log_data)
                x = np.linspace(min(survivors_list), max(survivors_list), 1000)
                try:
                    pdf_lognormal = stats.lognorm.pdf(x, s=sigma_log, scale=np.exp(mean_log))
                    plt.plot(x, pdf_lognormal, 'r--', linewidth=2, label='Log-normal approx.')
                except:
                    pass  # Skip if there's an error fitting the log-normal
        elif model_name == "induced":
            # Poisson approximation
            mean_val = np.mean(survivors_list)
            x = np.arange(max(0, int(mean_val) - 20), int(mean_val) + 20)
            try:
                pmf_poisson = stats.poisson.pmf(x, mean_val)
                plt.plot(x, pmf_poisson, 'r--', linewidth=2, label='Poisson approx.')
            except:
                pass  # Skip if there's an error fitting the Poisson
        
        plt.xlabel("Number of Resistant Survivors", fontsize=12)
        plt.ylabel("Probability Density", fontsize=12)
        title = "Probability Density Function"
        if model_name == "random":
            title += "\nExpected: Heavy-tailed distribution"
        elif model_name == "induced":
            title += "\nExpected: Poisson-like distribution"
        plt.title(title, fontsize=14)
        plt.legend()
        
        plt.tight_layout()
        save_path = os.path.join(results_path, f"luria_delbrueck_analysis_{model_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        # Create specialized visualization based on model type
        plt.figure(figsize=(10, 7))
        
        if model_name == "random":
            # Create a clear visual showing jackpot cultures
            plt.hist(survivors_list, bins=30, alpha=0.7, color='royalblue')
            plt.axvline(np.mean(survivors_list), color='red', linestyle='--', linewidth=2,
                      label=f"Mean = {np.mean(survivors_list):.2f}")
            plt.axvline(np.median(survivors_list), color='green', linestyle=':', linewidth=2,
                      label=f"Median = {np.median(survivors_list):.2f}")
            
            # Mark the "jackpot" cultures
            jackpot_threshold = np.mean(survivors_list) * 2
            jackpot_count = sum(1 for x in survivors_list if x > jackpot_threshold)
            jackpot_percent = (jackpot_count / len(survivors_list)) * 100
            
            plt.axvline(jackpot_threshold, color='orange', linestyle='-', linewidth=2,
                      label=f"Jackpot threshold")
            
            plt.xlabel("Number of Resistant Survivors", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.title(f"Random Model: Jackpot Cultures\n{jackpot_percent:.1f}% of cultures show jackpot effect", fontsize=16)
            plt.legend()
            
            # Add annotation
            plt.annotate("'Jackpot' cultures are a key signature\nof the Luria-Delbrück experiment", 
                      xy=(0.98, 0.98), xycoords='axes fraction', fontsize=12,
                      ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            jackpot_path = os.path.join(results_path, f"jackpot_culture_analysis_{model_name}.png")
            plt.savefig(jackpot_path, dpi=300)
            
        elif model_name == "induced":
            # Create a visual comparing to Poisson distribution
            plt.hist(survivors_list, bins=30, alpha=0.7, color='firebrick', density=True, label="Observed data")
            
            # Overlay Poisson PMF
            mean_val = np.mean(survivors_list)
            x = np.arange(max(0, int(mean_val) - 15), int(mean_val) + 15)
            try:
                pmf_poisson = stats.poisson.pmf(x, mean_val)
                plt.plot(x, pmf_poisson, 'b-', linewidth=3, label=f'Poisson (λ={mean_val:.2f})')
            except:
                pass
                
            plt.xlabel("Number of Resistant Survivors", fontsize=14)
            plt.ylabel("Probability", fontsize=14)
            plt.title(f"Induced Model: Poisson-like Distribution\nVMR = {vmr_val:.2f} (expected: ≈1 for Poisson)", fontsize=16)
            plt.legend()
            
            # Add annotation
            plt.annotate("Poisson-like distribution indicates\nmutations occur only in response to selection", 
                      xy=(0.98, 0.98), xycoords='axes fraction', fontsize=12,
                      ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            poisson_path = os.path.join(results_path, f"poisson_comparison_{model_name}.png")
            plt.savefig(poisson_path, dpi=300)
        
        plt.close()
        
        # Create a comparison with theoretical distributions
        plt.figure(figsize=(10, 6))
        
        # Generate theoretical data for appropriate comparison
        if model_name == "random":
            # For random model: compare with log-normal approximation
            nonzero_data = [x for x in survivors_list if x > 0]
            if len(nonzero_data) > 0:
                log_data = np.log(nonzero_data)
                mean_log = np.mean(log_data)
                sigma_log = np.std(log_data)
                theoretical_data = np.random.lognormal(mean_log, sigma_log, size=10000)
                plt.title("Random Model vs. Log-normal Distribution\n(approximation of Luria-Delbrück)", fontsize=16)
        elif model_name == "induced":
            # For induced model: compare with Poisson distribution
            theoretical_data = np.random.poisson(np.mean(survivors_list), size=10000)
            plt.title("Induced Model vs. Poisson Distribution", fontsize=16)
        else:
            # For combined model: show the actual distribution
            sns.kdeplot(survivors_list, color='green', linewidth=3, label="Combined Model")
            plt.title("Combined Model Distribution", fontsize=16)
            plt.xlabel("Number of Resistant Survivors", fontsize=14)
            plt.ylabel("Probability Density", fontsize=14)
            plt.legend()
            
            theory_path = os.path.join(results_path, f"theoretical_comparison_{model_name}.png")
            plt.savefig(theory_path, dpi=300)
            plt.close()
            return save_path
        
        # Plot both empirical and theoretical distributions
        sns.kdeplot(survivors_list, color='blue', linewidth=3, label=f"{model_name.capitalize()} Model" , bw_adjust=2)
        sns.kdeplot(theoretical_data, color='red', linewidth=2, linestyle='--', label="Theoretical Distribution" , bw_adjust=2)
        plt.xlabel("Number of Resistant Survivors", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)
        plt.legend()
        
        # Add annotation explaining the significance
        if model_name == "random":
            plt.annotate("Heavy-tailed distributions are expected for\nspontaneous mutations (Luria-Delbrück hypothesis)", 
                       xy=(0.02, 0.02), xycoords='axes fraction', fontsize=12,
                       ha='left', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        elif model_name == "induced":
            plt.annotate("Poisson-like distributions are expected for\ninduced mutations (variance ≈ mean)", 
                       xy=(0.02, 0.02), xycoords='axes fraction', fontsize=12,
                       ha='left', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        theory_path = os.path.join(results_path, f"theoretical_comparison_{model_name}.png")
        plt.savefig(theory_path, dpi=300)
        plt.close()
        
        return save_path
        
    except ImportError as e:
        log.warning(f"Visualization libraries not available: {e}")
        return None
    except Exception as e:
        log.error(f"Error creating visualizations: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        return None

@timing_decorator
def create_comparison_visualization(results_dict, results_path):
    """Create enhanced comparison visualizations for all models."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.gridspec import GridSpec
        from scipy import stats
        
        # Set style for better visualizations
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # Get model names and their corresponding data
        model_names = list(results_dict.keys())
        
        # Calculate key statistics for all models
        stats_dict = {}
        for model in model_names:
            data = results_dict[model]
            mean_val = np.mean(data)
            var_val = np.var(data)
            std_val = np.std(data)
            vmr_val = var_val / mean_val if mean_val > 0 else 0
            cv_val = std_val / mean_val if mean_val > 0 else 0
            
            stats_dict[model] = {
                'mean': mean_val,
                'variance': var_val,
                'std_dev': std_val,
                'vmr': vmr_val,
                'cv': cv_val
            }
        
        # Create figure with custom grid layout
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(2, 2, figure=fig)
        
        # Define colors and display names
        colors = {'random': 'royalblue', 'induced': 'firebrick', 'combined': 'forestgreen'}
        model_display_names = {'random': 'Random', 'induced': 'Induced', 'combined': 'Combined'}
        
        # 1. DIRECT COMPARISON PLOT (top left) - Shows actual distributions on same scale
        ax1 = fig.add_subplot(gs[0, 0])
        
        for model in model_names:
            # Use kernel density estimation for smooth distributions
            sns.kdeplot(results_dict[model], ax=ax1, color=colors[model], 
                      label=f"{model_display_names[model]} (VMR={stats_dict[model]['vmr']:.2f})", 
                      linewidth=2.5)
        
        # Set common labels and title
        ax1.set_xlabel("Number of Resistant Survivors", fontsize=12)
        ax1.set_ylabel("Density", fontsize=12)
        ax1.set_title("Distribution Comparison (Same Scale)", fontweight='bold', fontsize=14)
        ax1.legend(loc='best')
        
        # Add vertical lines at means
        for model in model_names:
            ax1.axvline(stats_dict[model]['mean'], color=colors[model], linestyle='--', alpha=0.7)
            
        # Add annotation explaining the key difference
        if 'random' in model_names and 'induced' in model_names:
            ax1.annotate("Note different scales and shapes:\nRandom model has wider spread\nInduced model is more concentrated", 
                       xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10,
                       ha='left', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 2. LOG-LOG CCDF PLOT (top right) - Best for showing Luria-Delbrück effect
        ax2 = fig.add_subplot(gs[0, 1])
        
        for model in model_names:
            # Calculate CCDF
            sorted_data = np.sort(results_dict[model])
            ccdf = 1 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            # Plot on log-log scale
            ax2.loglog(sorted_data, ccdf, color=colors[model], 
                     label=f"{model_display_names[model]} Model", linewidth=2.5)
        
        ax2.set_xlabel("Number of Resistant Survivors (log scale)", fontsize=12)
        ax2.set_ylabel("P(X > x) (log scale)", fontsize=12)
        ax2.set_title("Log-Log CCDF Plot - Key for Luria-Delbrück Pattern", fontweight='bold', fontsize=14)
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc='best')
        
        # Add annotation explaining the significance
        ax2.annotate("Straight line indicates power-law behavior\n(characteristic of Luria-Delbrück)\nCurved line indicates Poisson-like behavior", 
                   xy=(0.02, 0.02), xycoords='axes fraction', fontsize=10,
                   ha='left', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 3. VARIANCE-TO-MEAN RATIO COMPARISON (bottom left) - Critical for hypothesis testing
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Extract VMR values
        vmr_values = [stats_dict[model]['vmr'] for model in model_names]
        
        # Create bar chart with custom colors
        bars = ax3.bar([model_display_names[m] for m in model_names], vmr_values, 
                     color=[colors[m] for m in model_names], alpha=0.7, edgecolor='black', width=0.6)
        
        # Set labels and title
        ax3.set_xlabel("Model", fontsize=12)
        ax3.set_ylabel("Variance-to-Mean Ratio (VMR)", fontsize=12)
        ax3.set_title("Variance-to-Mean Ratio Comparison\nKey Indicator for Luria-Delbrück Effect", 
                    fontweight='bold', fontsize=14)
        
        # Add horizontal reference line at VMR=1 (Poisson expectation)
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.7, 
                  label='Poisson Distribution (VMR = 1)')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                y_pos = height + 0.1
                ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Customize y-limits for better visualization
        ax3.set_ylim(0, max(vmr_values) * 1.2)
        
        # Add annotation explaining VMR significance
        ax3.annotate("VMR ≈ 1: Poisson distribution (expected for induced mutations)\nVMR >> 1: Heavy-tailed distribution (expected for random mutations)", 
                   xy=(0.5, 0.02), xycoords='axes fraction', fontsize=10, ha='center',
                   va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add legend
        ax3.legend()
        
        # 4. SUMMARY VISUALIZATION (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Create table with key statistics
        stats_rows = []
        headers = ["Statistic"] + [model_display_names[m] for m in model_names]
        
        # Add rows of key statistics
        stats_rows.append(["Mean"] + [f"{stats_dict[m]['mean']:.2f}" for m in model_names])
        stats_rows.append(["Variance"] + [f"{stats_dict[m]['variance']:.2f}" for m in model_names])
        stats_rows.append(["VMR"] + [f"{stats_dict[m]['vmr']:.2f}" for m in model_names])
        
        # Calculate ratio between random and induced VMR if both exist
        if 'random' in model_names and 'induced' in model_names:
            random_vmr = stats_dict['random']['vmr']
            induced_vmr = stats_dict['induced']['vmr']
            vmr_ratio = random_vmr / induced_vmr if induced_vmr > 0 else float('inf')
            stats_rows.append(["Random/Induced VMR Ratio", f"{vmr_ratio:.2f}", "", ""])
        
        # Create the table
        table = ax4.table(cellText=stats_rows, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.8)
        
        # Customize table appearance
        for i in range(len(stats_rows) + 1):  # +1 for header
            for j in range(len(headers)):
                cell = table[(i, j)]
                # Header row
                if i == 0:
                    cell.set_facecolor('#D8D8D8')
                    cell.set_text_props(weight='bold')
                # First column (statistic names)
                if j == 0:
                    cell.set_text_props(weight='bold')
                # VMR row - highlight values
                if i == 3:
                    if j > 0:
                        model_idx = j - 1
                        model = model_names[model_idx]
                        vmr = stats_dict[model]['vmr']
                        if vmr > 1.1:  # Well above 1 (random mutation)
                            cell.set_facecolor('#BBFFBB')  # Green
                        elif 0.9 <= vmr <= 1.1:  # Close to 1 (induced mutation)
                            cell.set_facecolor('#FFFFBB')  # Yellow
        
        # Hide axes for table
        ax4.axis('off')
        ax4.set_title("Key Statistics", fontweight='bold', fontsize=14)
        
        # Add conclusion based on results
        if 'random' in stats_dict and 'induced' in stats_dict:
            random_vmr = stats_dict['random']['vmr']
            induced_vmr = stats_dict['induced']['vmr']
            
            if random_vmr > induced_vmr * 1.5:
                conclusion = "CONCLUSION: The Random model shows significantly higher VMR than the Induced model,\nstrongly supporting the Luria-Delbrück hypothesis of spontaneous mutations."
                color = 'green'
            elif random_vmr > induced_vmr:
                conclusion = "CONCLUSION: The Random model shows moderately higher VMR than the Induced model,\nsupporting the Luria-Delbrück hypothesis of spontaneous mutations."
                color = 'darkgreen'
            else:
                conclusion = "CONCLUSION: Results inconclusive - Random model VMR is not higher than Induced model VMR.\nConsider adjusting simulation parameters to better demonstrate the Luria-Delbrück effect."
                color = 'red'
                
            ax4.annotate(conclusion, xy=(0.5, 0.05), xycoords='axes fraction', fontsize=12,
                       ha='center', va='bottom', color=color, weight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        # Save the overall comparison figure
        save_path = os.path.join(results_path, "enhanced_model_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional specialized comparison plots
        
        # 1. DIRECT COMPARISON IN SINGLE PLOT
        plt.figure(figsize=(10, 7))
        
        for model in model_names:
            sns.kdeplot(results_dict[model], color=colors[model], 
                      label=f"{model_display_names[model]} (VMR={stats_dict[model]['vmr']:.2f})",
                      linewidth=2.5)
        
        plt.xlabel("Number of Resistant Survivors", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.title("Distribution Comparison of Mutation Models", fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Add vertical lines at means
        for model in model_names:
            plt.axvline(stats_dict[model]['mean'], color=colors[model], linestyle='--', alpha=0.7)
        
        # Add annotation about the key difference
        if 'random' in model_names and 'induced' in model_names:
            plt.annotate("Key Luria-Delbrück Signature:\nRandom model shows wider, skewed distribution ('jackpot' effect)\nInduced model shows narrower, more symmetric distribution", 
                       xy=(0.98, 0.02), xycoords='axes fraction', fontsize=12,
                       ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        single_plot_path = os.path.join(results_path, "direct_comparison_single_plot.png")
        plt.savefig(single_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. VMR BAR CHART WITH RATIO
        plt.figure(figsize=(10, 7))
        
        # First plot the VMR values
        bars = plt.bar([model_display_names[m] for m in model_names], 
                     [stats_dict[m]['vmr'] for m in model_names],
                     color=[colors[m] for m in model_names], alpha=0.7, width=0.6)
        
        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add horizontal line at VMR=1
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, 
                  label='Poisson Distribution (VMR = 1)')
        
        plt.xlabel('Mutation Model', fontsize=14)
        plt.ylabel('Variance-to-Mean Ratio (VMR)', fontsize=14)
        plt.title('Variance-to-Mean Ratio: Key Indicator for Luria-Delbrück Effect', 
               fontsize=16, fontweight='bold')
        
        # Add ratio indicator if both random and induced models exist
        if 'random' in stats_dict and 'induced' in stats_dict:
            random_vmr = stats_dict['random']['vmr']
            induced_vmr = stats_dict['induced']['vmr']
            vmr_ratio = random_vmr / induced_vmr if induced_vmr > 0 else float('inf')
            
            plt.annotate(f"Random/Induced VMR Ratio: {vmr_ratio:.2f}\n"
                       f"{'✓ SUPPORTS' if vmr_ratio > 1 else '✗ DOES NOT SUPPORT'} Luria-Delbrück hypothesis", 
                       xy=(0.5, 0.02), xycoords='axes fraction', fontsize=14, ha='center',
                       va='bottom', color='green' if vmr_ratio > 1 else 'red', weight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.ylim(0, max([stats_dict[m]['vmr'] for m in model_names]) * 1.2)
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend()
        
        vmr_chart_path = os.path.join(results_path, "vmr_comparison_chart.png")
        plt.savefig(vmr_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create facet grid of histograms for clearer comparison
        plt.figure(figsize=(15, 5))
        
        if len(model_names) > 1:
            # Prepare data for side-by-side plotting
            all_data = []
            for model in model_names:
                model_data = results_dict[model]
                model_df = pd.DataFrame({
                    'survivors': model_data,
                    'model': model_display_names[model]
                })
                all_data.append(model_df)
            
            combined_df = pd.concat(all_data)
            
            # Create facet grid with explicitly scaled subplots
            g = sns.FacetGrid(combined_df, col="model", height=4, aspect=1.2,
                             sharex=False, sharey=False)
            
            # Add histograms with density curves
            g.map_dataframe(sns.histplot, x="survivors", kde=True)
            
            # Set axis labels and titles
            g.set_axis_labels("Number of Resistant Survivors", "Frequency")
            g.set_titles(col_template="{col_name} Model")
            
            # Add VMR annotations to each subplot
            for i, model in enumerate(model_names):
                g.axes[0, i].annotate(f"VMR = {stats_dict[model]['vmr']:.2f}",
                                   xy=(0.5, 0.95), xycoords='axes fraction',
                                   ha='center', va='top', fontsize=12,
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # Add explanatory notes based on model type
                if model == 'random':
                    g.axes[0, i].annotate("Wide distribution with\nlong right tail\n(Luria-Delbrück signature)",
                                      xy=(0.05, 0.05), xycoords='axes fraction',
                                      ha='left', va='bottom', fontsize=10,
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                elif model == 'induced':
                    g.axes[0, i].annotate("Narrower, more\nsymmetrical distribution\n(Poisson-like)",
                                      xy=(0.05, 0.05), xycoords='axes fraction',
                                      ha='left', va='bottom', fontsize=10,
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            g.tight_layout()
            
            # Save histogram comparison
            hist_path = os.path.join(results_path, "clear_histogram_comparison.png")
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create a summary figure specifically designed for presentation
        plt.figure(figsize=(12, 8))
        
        if 'random' in model_names and 'induced' in model_names:
            # Calculate key statistics
            random_vmr = stats_dict['random']['vmr']
            induced_vmr = stats_dict['induced']['vmr']
            vmr_ratio = random_vmr / induced_vmr if induced_vmr > 0 else float('inf')
            
            # Plot simplified distributions
            plt.plot([], [], color=colors['random'], linewidth=3, label=f"Random Model (VMR={random_vmr:.2f})")
            plt.plot([], [], color=colors['induced'], linewidth=3, label=f"Induced Model (VMR={induced_vmr:.2f})")
            
            # Create a textbox with key findings
            findings_text = (
                f"KEY FINDINGS:\n\n"
                f"1. Random Model VMR: {random_vmr:.2f}\n"
                f"   - {'MATCHES' if random_vmr > 1.1 else 'DOES NOT MATCH'} Luria-Delbrück prediction (VMR > 1)\n\n"
                f"2. Induced Model VMR: {induced_vmr:.2f}\n"
                f"   - {'MATCHES' if 0.9 <= induced_vmr <= 1.1 else 'DOES NOT MATCH'} Poisson prediction (VMR ≈ 1)\n\n"
                f"3. VMR Ratio (Random/Induced): {vmr_ratio:.2f}\n"
                f"   - {'SUPPORTS' if vmr_ratio > 1.5 else 'WEAKLY SUPPORTS' if vmr_ratio > 1 else 'DOES NOT SUPPORT'} Luria-Delbrück hypothesis\n\n"
                f"CONCLUSION:\n"
                f"{'The simulation results strongly support the Luria-Delbrück hypothesis that mutations occur spontaneously during growth rather than as a response to selection pressure.' if vmr_ratio > 1.5 else 'The simulation results weakly support the Luria-Delbrück hypothesis. Consider adjusting parameters for clearer results.' if vmr_ratio > 1 else 'The simulation results do not clearly support the Luria-Delbrück hypothesis. Consider adjusting simulation parameters.'}"
            )
            
            plt.text(0.5, 0.5, findings_text, ha='center', va='center', fontsize=14,
                   bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=1',
                           edgecolor='gray'), transform=plt.gca().transAxes)
            
            plt.axis('off')
            plt.title("Luria-Delbrück Experiment: Summary of Findings", fontsize=18, fontweight='bold')
            plt.legend(loc='upper center', fontsize=12)
            
            summary_path = os.path.join(results_path, "experiment_summary.png")
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return save_path
    
    except ImportError as e:
        log.warning(f"Visualization libraries not available: {e}")
        return None
    except Exception as e:
        log.error(f"Error creating comparison visualizations: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        return None

# Additional helper function for creating presentations
def create_presentation_visualizations(results_dict, results_path):
    """Create visualizations specifically designed for presentations about Luria-Delbrück experiment"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        
        # Set style for better visualizations
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        
        # Get model names and their corresponding data
        model_names = list(results_dict.keys())
        
        # Calculate key statistics
        stats_dict = {}
        for model in model_names:
            data = results_dict[model]
            mean_val = np.mean(data)
            var_val = np.var(data)
            vmr_val = var_val / mean_val if mean_val > 0 else 0
            
            stats_dict[model] = {
                'mean': mean_val,
                'variance': var_val,
                'vmr': vmr_val
            }
        
        # Define colors and display names
        colors = {'random': 'royalblue', 'induced': 'firebrick', 'combined': 'forestgreen'}
        model_display_names = {'random': 'Random', 'induced': 'Induced', 'combined': 'Combined'}
        
        # 1. THEORY SLIDE: Theoretical Assumptions
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        
        title = "Luria-Delbrück Experiment: Theoretical Assumptions"
        text = (
            "Two competing hypotheses of bacterial mutation:\n\n"
            "1. RANDOM MUTATION (Darwinian):\n"
            "   • Mutations occur spontaneously during growth\n"
            "   • Independent of selective pressure\n"
            "   • Some cultures will have early mutations leading to many resistant cells ('jackpots')\n"
            "   • Leads to highly variable outcomes (high variance-to-mean ratio)\n\n"
            "2. INDUCED MUTATION (Lamarckian):\n"
            "   • Mutations occur only in response to selective pressure\n"
            "   • All bacteria have equal probability of mutation\n"
            "   • Leads to Poisson distribution of resistant cells\n"
            "   • Variance approximately equals mean (variance-to-mean ratio ≈ 1)\n\n"
            "The distribution pattern of resistant cells across cultures can\n"
            "distinguish between these two hypotheses."
        )
        
        plt.text(0.5, 0.95, title, ha='center', va='top', fontsize=20, fontweight='bold')
        plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=16)
        
        theory_path = os.path.join(results_path, "presentation_theory.png")
        plt.savefig(theory_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. DISTRIBUTIONS SLIDE: Distribution Patterns
        if 'random' in model_names and 'induced' in model_names:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            
            # Random model distribution
            sns.histplot(results_dict['random'], kde=True, color=colors['random'], ax=axes[0])
            axes[0].set_title(f"Random Model (VMR = {stats_dict['random']['vmr']:.2f})", fontsize=16)
            axes[0].set_xlabel("Number of Resistant Survivors")
            axes[0].set_ylabel("Frequency")
            
            # Add annotation for random model
            axes[0].annotate("Wide distribution with\nlong right tail\n('jackpot' effect)",
                         xy=(0.95, 0.95), xycoords='axes fraction',
                         ha='right', va='top', fontsize=14,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Induced model distribution
            sns.histplot(results_dict['induced'], kde=True, color=colors['induced'], ax=axes[1])
            axes[1].set_title(f"Induced Model (VMR = {stats_dict['induced']['vmr']:.2f})", fontsize=16)
            axes[1].set_xlabel("Number of Resistant Survivors")
            axes[1].set_ylabel("Frequency")
            
            # Add annotation for induced model
            axes[1].annotate("Narrower, more\nsymmetrical distribution\n(Poisson-like)",
                         xy=(0.95, 0.95), xycoords='axes fraction',
                         ha='right', va='top', fontsize=14,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            plt.suptitle("Distribution Patterns of Resistant Cells", fontsize=18, fontweight='bold')
            plt.tight_layout()
            
            distributions_path = os.path.join(results_path, "presentation_distributions.png")
            plt.savefig(distributions_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. VMR SLIDE: Variance-to-Mean Ratio Comparison
        plt.figure(figsize=(12, 8))
        
        # Extract VMR values
        vmr_values = [stats_dict[model]['vmr'] for model in model_names]
        
        # Create bar chart with custom colors
        bars = plt.bar([model_display_names[m] for m in model_names], vmr_values, 
                     color=[colors[m] for m in model_names], alpha=0.8, width=0.6)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Add horizontal line at VMR=1
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, 
                  label='Poisson Expectation (VMR = 1)')
        
        plt.xlabel('Mutation Model', fontsize=16)
        plt.ylabel('Variance-to-Mean Ratio (VMR)', fontsize=16)
        plt.title('Variance-to-Mean Ratio Comparison\nKey Statistical Evidence for Luria-Delbrück Hypothesis', 
               fontsize=18, fontweight='bold')
        
        # Add explanation text
        plt.annotate("VMR ≈ 1: Poisson distribution (expected for induced mutations)\nVMR > 1: Indicates non-random clustering ('jackpot' cultures)", 
                   xy=(0.5, 0.05), xycoords='axes fraction', fontsize=14, ha='center',
                   va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.ylim(0, max(vmr_values) * 1.3)
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(fontsize=12)
        
        vmr_slide_path = os.path.join(results_path, "presentation_vmr.png")
        plt.savefig(vmr_slide_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. CONCLUSION SLIDE: Results and Interpretation
        if 'random' in model_names and 'induced' in model_names:
            plt.figure(figsize=(12, 8))
            plt.axis('off')
            
            # Calculate random/induced ratio
            random_vmr = stats_dict['random']['vmr']
            induced_vmr = stats_dict['induced']['vmr']
            vmr_ratio = random_vmr / induced_vmr if induced_vmr > 0 else float('inf')
            
            title = "Luria-Delbrück Experiment: Results and Interpretation"
            
            conclusion_color = 'green' if vmr_ratio > 1.5 else 'orange' if vmr_ratio > 1 else 'red'
            
            if vmr_ratio > 1.5:
                conclusion = "The simulation results STRONGLY SUPPORT the Luria-Delbrück hypothesis"
            elif vmr_ratio > 1:
                conclusion = "The simulation results SUPPORT the Luria-Delbrück hypothesis"
            else:
                conclusion = "The simulation results do not clearly support the Luria-Delbrück hypothesis"
            
            text = (
                f"KEY FINDINGS:\n\n"
                f"1. Random Model VMR: {random_vmr:.2f}\n"
                f"   • {'MATCHES' if random_vmr > 1.1 else 'DOES NOT MATCH'} Luria-Delbrück prediction (VMR > 1)\n\n"
                f"2. Induced Model VMR: {induced_vmr:.2f}\n"
                f"   • {'MATCHES' if 0.9 <= induced_vmr <= 1.1 else 'DOES NOT MATCH'} Poisson prediction (VMR ≈ 1)\n\n"
                f"3. VMR Ratio (Random/Induced): {vmr_ratio:.2f}\n\n"
                f"CONCLUSION:\n\n"
                f"{conclusion}\n\n"
                f"These results indicate that bacterial mutations occur\n"
                f"spontaneously during growth rather than as a direct\n"
                f"response to selection pressure, supporting the\n"
                f"Darwinian view of evolution rather than the Lamarckian view."
            )
            
            plt.text(0.5, 0.95, title, ha='center', va='top', fontsize=20, fontweight='bold')
            plt.text(0.5, 0.8, conclusion, ha='center', va='top', fontsize=18, 
                   fontweight='bold', color=conclusion_color)
            plt.text(0.5, 0.4, text, ha='center', va='center', fontsize=16)
            
            conclusion_path = os.path.join(results_path, "presentation_conclusion.png")
            plt.savefig(conclusion_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Return paths to all created slides
        return {
            'theory': theory_path,
            'distributions': locals().get('distributions_path'),
            'vmr': vmr_slide_path,
            'conclusion': locals().get('conclusion_path')
        }
    
    except ImportError as e:
        log.warning(f"Visualization libraries not available: {e}")
        return None
    except Exception as e:
        log.error(f"Error creating presentation visualizations: {str(e)}")
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
    
    print(f"Returning from run_all_models: {len(results)} results, {len(stats)} stats, {len(elapsed_times)} times")
    return results, stats, elapsed_times  # This should fix the "not enough values to unpack" error

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