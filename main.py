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
# Simulation functions using the Organism class
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
        
        # Set style for better visualizations
        sns.set_style("whitegrid")
        
        # Create figure and axes
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Luria-Delbrück Experiment: {model_name.capitalize()} Model", fontsize=16)
        
        # 1. Histogram of survivors
        axes[0, 0].hist(survivors_list, bins='auto', alpha=0.7, color='royalblue')
        axes[0, 0].set_xlabel("Number of Resistant Survivors")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Distribution of Resistant Survivors")
        
        # 2. Log-scale histogram (key for Luria-Delbrück analysis)
        if max(survivors_list) > 0:
            bins = np.logspace(0, np.log10(max(survivors_list) + 1), 20)
            axes[0, 1].hist(survivors_list, bins=bins, alpha=0.7, color='forestgreen')
            axes[0, 1].set_xscale('log')
            axes[0, 1].set_xlabel("Number of Resistant Survivors (log scale)")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_title("Log-Scale Distribution")
        
        # 3. Cumulative distribution
        sorted_data = np.sort(survivors_list)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 0].step(sorted_data, cumulative, where='post', color='darkorange')
        axes[1, 0].set_xlabel("Number of Resistant Survivors")
        axes[1, 0].set_ylabel("Cumulative Probability")
        axes[1, 0].set_title("Cumulative Distribution")
        
        # 4. Box plot
        axes[1, 1].boxplot(survivors_list, vert=False, patch_artist=True, 
                         boxprops=dict(facecolor='lightblue'))
        axes[1, 1].set_xlabel("Number of Resistant Survivors")
        axes[1, 1].set_title("Box Plot")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure
        save_path = os.path.join(results_path, f"analysis_{model_name}.png")
        plt.savefig(save_path, dpi=300)
        log.info(f"Visualization saved to {save_path}")
        
        # Optional: display if in interactive mode
        plt.close()
        
        return save_path
    except ImportError as e:
        log.warning(f"Visualization libraries not available: {e}")
        return None
    except Exception as e:
        log.error(f"Error creating visualizations: {str(e)}")
        return None

@timing_decorator
def create_comparison_visualization(results_dict, results_path):
    """Create comparison visualizations for all models."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style for better visualizations
        sns.set_style("whitegrid")
        
        # Get model names and their corresponding data
        model_names = list(results_dict.keys())
        
        # Create figure and axes
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Luria-Delbrück Experiment: Model Comparison", fontsize=16)
        
        # 1. Histogram comparison
        for model in model_names:
            sns.histplot(results_dict[model], 
                         kde=True, 
                         label=model.capitalize(),
                         alpha=0.5,
                         ax=axes[0, 0])
        axes[0, 0].set_xlabel("Number of Resistant Survivors")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Distribution Comparison")
        axes[0, 0].legend()
        
        # 2. Box plot comparison
        data_for_boxplot = []
        labels_for_boxplot = []
        for model in model_names:
            data_for_boxplot.append(results_dict[model])
            labels_for_boxplot.append(model.capitalize())
        
        axes[0, 1].boxplot(data_for_boxplot, 
                         vert=True, 
                         patch_artist=True,
                         labels=labels_for_boxplot)
        axes[0, 1].set_ylabel("Number of Resistant Survivors")
        axes[0, 1].set_title("Box Plot Comparison")
        
        # 3. Coefficient of Variation Comparison
        cv_values = []
        for model in model_names:
            survivors = results_dict[model]
            mean = np.mean(survivors)
            std = np.std(survivors)
            cv = std / mean if mean > 0 else 0
            cv_values.append(cv)
        
        bars = axes[1, 0].bar(model_names, cv_values, color=['royalblue', 'forestgreen', 'darkorange'])
        axes[1, 0].set_xlabel("Model")
        axes[1, 0].set_ylabel("Coefficient of Variation")
        axes[1, 0].set_title("Coefficient of Variation Comparison")
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 4. Summary statistics table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        # Create table data
        stats = []
        headers = ["Statistic"] + [m.capitalize() for m in model_names]
        
        # Add rows of statistics
        stats.append(["Mean"] + [f"{np.mean(results_dict[m]):.2f}" for m in model_names])
        stats.append(["Variance"] + [f"{np.var(results_dict[m]):.2f}" for m in model_names])
        stats.append(["Std Dev"] + [f"{np.std(results_dict[m]):.2f}" for m in model_names])
        stats.append(["Median"] + [f"{np.median(results_dict[m]):.2f}" for m in model_names])
        stats.append(["Max"] + [f"{np.max(results_dict[m])}" for m in model_names])
        stats.append(["CV"] + [f"{cv_values[i]:.2f}" for i in range(len(model_names))])
        stats.append(["Zero Count"] + [f"{sum(1 for x in results_dict[m] if x == 0)}" for m in model_names])
        
        table = axes[1, 1].table(cellText=stats, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure
        save_path = os.path.join(results_path, "model_comparison.png")
        plt.savefig(save_path, dpi=300)
        log.info(f"Comparison visualization saved to {save_path}")
        
        plt.close()
        
        return save_path
    except ImportError as e:
        log.warning(f"Visualization libraries not available: {e}")
        return None
    except Exception as e:
        log.error(f"Error creating comparison visualizations: {str(e)}")
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
    
    _, vis_time = create_visualizations(random_survivors, 'random', args.results_path)
    elapsed_times['random_visualization'] = vis_time
    
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
    
    _, vis_time = create_visualizations(induced_survivors, 'induced', args.results_path)
    elapsed_times['induced_visualization'] = vis_time
    
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
    
    _, vis_time = create_visualizations(combined_survivors, 'combined', args.results_path)
    elapsed_times['combined_visualization'] = vis_time
    
    # 4. Create comparison visualizations
    log.info("\n" + "="*50)
    log.info("Creating Comparison Visualizations")
    log.info("="*50)
    
    _, comparison_time = create_comparison_visualization(results, args.results_path)
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

def create_time_summary(elapsed_times, results_path):
    """Create a summary of execution times for all operations"""
    summary_path = os.path.join(results_path, "execution_time_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Execution Time Summary\n")
        f.write("="*80 + "\n\n")
        
        # Group times by category
        categories = {
            "Simulation": ["random", "induced", "combined"],
            "Analysis": ["random_analysis", "induced_analysis", "combined_analysis"],
            "Visualization": ["random_visualization", "induced_visualization", "combined_visualization", "comparison_visualization"],
            "Other": ["report_creation"]
        }
        
        # Calculate total time
        total_time = sum(elapsed_times.values())
        
        # Write summary by category
        for category, keys in categories.items():
            f.write(f"{category} Times:\n")
            f.write("-"*40 + "\n")
            category_total = 0
            
            for key in keys:
                if key in elapsed_times:
                    time_value = elapsed_times[key]
                    percent = (time_value / total_time) * 100
                    f.write(f"{key:<25}: {time_value:.2f} seconds ({percent:.2f}%)\n")
                    category_total += time_value
            
            cat_percent = (category_total / total_time) * 100
            f.write(f"{'Total ' + category:<25}: {category_total:.2f} seconds ({cat_percent:.2f}%)\n\n")
        
        # Write overall total
        f.write("="*40 + "\n")
        f.write(f"Total Execution Time: {total_time:.2f} seconds\n")
        f.write("="*40 + "\n")
    
    log.info(f"Execution time summary created at {summary_path}")
    return summary_path

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
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

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
            # Profile the all-models run
            result, profiler = run_profiling(run_all_models, args)
            save_profiling_results(profiler, args.results_path, "all_models_profile")
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
            
            # Run with profiling
            (survivors_list, _), profiler = run_profiling(func, **kwargs)
            save_profiling_results(profiler, args.results_path, f"{args.model}_profile")
            
            # Continue with analysis and visualization
            stats = analyze_results(survivors_list, args.model)[0]
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
            print(f"{model.capitalize()} Model:")
            print(f"  Mean # survivors: {model_stats['mean']:.2f}")
            print(f"  Variance: {model_stats['variance']:.2f}")
            print(f"  Coefficient of Variation: {model_stats['coefficient_of_variation']:.2f}")
            print(f"  Simulation time: {times[model]:.2f} seconds")
            print("-" * 40)
        
        # Indicate where results are saved
        print(f"\nDetailed results and visualizations saved to:")
        print(f"  {args.results_path}")
        
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
        
        # Also print to console
        print(f"\nModel: {args.model}")
        print(f"Number of cultures: {args.num_cultures}")
        print(f"Mean # survivors: {stats['mean']:.4f}")
        print(f"Variance of # survivors: {stats['variance']:.4f}")
        print(f"Coefficient of variation: {stats['coefficient_of_variation']:.4f}")
        print(f"Cultures with zero resistant organisms: {stats['zero_resistant_percent']:.2f}%")
        print(f"Simulation time: {elapsed_time:.2f} seconds")
        print(f"Analysis time: {stats_time:.2f} seconds")
        
        # Create visualizations
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