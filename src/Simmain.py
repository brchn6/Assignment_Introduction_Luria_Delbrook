import numpy as np
import argparse
import os
import sys
import logging
import time
import multiprocessing as mp
from datetime import datetime
from functools import partial
from tqdm import tqdm

# Setup logging
def init_log(results_path, log_level="INFO"):
    """Initialize logging with specified log level"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    
    path = os.path.join(results_path, 'luria_delbruck_simulation.log')
    fh = logging.FileHandler(path, mode='w')
    fh.setLevel(level)
    logging.getLogger().addHandler(fh)

    logging.info("Logging initialized successfully.")
    return logging, results_path

# Time tracking decorator
def timing_decorator(func):
    """Decorator to measure and log execution time of functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Function {func.__name__} took {elapsed_time:.2f} seconds to execute")
        return result, elapsed_time
    return wrapper

# Organism class
class Organism:
    """Represents a single organism in the simulation with a genome that can mutate"""
    N_GENES = 1  # Default value, will be overridden by command line arg
    P_MUTATION = 0.1  # Default, overridden in main
    
    def __init__(self, genome=None):
        # Initialize with non-resistant genome (all zeros) by default
        self.genome = genome if genome is not None else np.zeros(Organism.N_GENES, dtype=int)
        self.is_resistant = bool(self.genome[0])  # First gene determines resistance

    def mutate(self, mutation_rate=None):
        """Apply random mutations to the genome based on mutation probability"""
        rate = mutation_rate if mutation_rate is not None else Organism.P_MUTATION
        
        # Check for mutations in each gene
        for i in range(Organism.N_GENES):
            if np.random.rand() < rate:
                self.genome[i] = 1 - self.genome[i]  # Flip bit
        
        # Update resistance status
        self.is_resistant = bool(self.genome[0])
        return self

    def reproduce(self, mutation_rate=None):
        """Create an offspring with possible mutations"""
        offspring = Organism(genome=self.genome.copy())
        offspring.mutate(mutation_rate)
        return offspring

# Simulation Functions
def process_culture_random(culture_id, initial_population_size, num_generations, mutation_rate):
    """Process a single culture for random mutations model"""
    # Initialize population with sensitive organisms
    population = [Organism() for _ in range(initial_population_size)]
    
    # Simulate generations
    for _ in range(num_generations):
        new_population = []
        for organism in population:
            # Each organism produces two offspring
            offspring1 = organism.reproduce(mutation_rate)
            offspring2 = organism.reproduce(mutation_rate)
            
            new_population.append(offspring1)
            new_population.append(offspring2)
            
        population = new_population
    
    # Count survivors (resistant organisms) after all generations
    survivors = sum(org.is_resistant for org in population)
    return survivors

def process_culture_induced(culture_id, initial_population_size, num_generations, mutation_rate):
    """Process a single culture for induced mutations model"""
    # Initialize population with sensitive organisms
    population = [Organism() for _ in range(initial_population_size)]
    
    # Simulate generations with no mutations
    for _ in range(num_generations):
        new_population = []
        for organism in population:
            # Each organism reproduces without mutation
            offspring1 = Organism(genome=organism.genome.copy())
            offspring2 = Organism(genome=organism.genome.copy())
            
            new_population.append(offspring1)
            new_population.append(offspring2)
            
        population = new_population
    
    # Apply induced mutations at the end
    for organism in population:
        organism.mutate(mutation_rate)
    
    # Count survivors
    survivors = sum(org.is_resistant for org in population)
    return survivors

def process_culture_combined(culture_id, initial_population_size, num_generations, random_mut_rate, induced_mut_rate):
    """Process a single culture for combined mutations model"""
    # Initialize population with sensitive organisms
    population = [Organism() for _ in range(initial_population_size)]
    
    # Simulate generations with random mutations
    for _ in range(num_generations):
        new_population = []
        for organism in population:
            # Each organism produces two offspring with possible mutations
            offspring1 = organism.reproduce(random_mut_rate)
            offspring2 = organism.reproduce(random_mut_rate)
            
            new_population.append(offspring1)
            new_population.append(offspring2)
            
        population = new_population
    
    # Apply additional induced mutations at the end
    for organism in population:
        organism.mutate(induced_mut_rate)
    
    # Count survivors
    survivors = sum(org.is_resistant for org in population)
    return survivors

@timing_decorator
def simulate_model(model_type, num_cultures, initial_population_size, num_generations, 
                  mutation_rate, induced_mutation_rate=None, use_parallel=True, n_processes=None):
    """Generic simulation function that handles all model types"""
    survivors_list = []
    
    # Select the appropriate process function based on model type
    if model_type == "random":
        process_func = partial(process_culture_random, 
                              initial_population_size=initial_population_size,
                              num_generations=num_generations,
                              mutation_rate=mutation_rate)
    elif model_type == "induced":
        process_func = partial(process_culture_induced, 
                              initial_population_size=initial_population_size,
                              num_generations=num_generations,
                              mutation_rate=mutation_rate)
    else:  # combined
        process_func = partial(process_culture_combined, 
                              initial_population_size=initial_population_size,
                              num_generations=num_generations,
                              random_mut_rate=mutation_rate,
                              induced_mut_rate=induced_mutation_rate)
    
    if use_parallel and num_cultures > 10:
        # Use multiprocessing for better performance
        n_processes = n_processes or max(1, mp.cpu_count() - 1)
        logging.info(f"Using parallel processing with {n_processes} processes")
        
        # Process cultures in parallel
        with mp.Pool(processes=n_processes) as pool:
            survivors_list = list(tqdm(
                pool.imap(process_func, range(num_cultures)),
                total=num_cultures,
                desc=f"{model_type.capitalize()} Model Progress",
                unit="culture"
            ))
    else:
        # Sequential processing with progress bar
        for culture_id in tqdm(range(num_cultures), 
                              desc=f"{model_type.capitalize()} Model Progress", 
                              unit="culture"):
            survivors = process_func(culture_id)
            survivors_list.append(survivors)
    
    return survivors_list

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
    
    return results

def create_visualization(survivors_list, model_name, results_path):
    """Create basic visualizations of the simulation results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create a simple figure with histogram and log-scale distribution
        plt.figure(figsize=(12, 5))
        
        # Regular histogram
        plt.subplot(1, 2, 1)
        plt.hist(survivors_list, bins='auto', alpha=0.7, color='royalblue')
        plt.xlabel("Number of Resistant Survivors")
        plt.ylabel("Frequency")
        plt.title(f"{model_name.capitalize()} Model Distribution")
        
        # Log-scale histogram (key for Luria-Delbrück analysis)
        plt.subplot(1, 2, 2)
        if max(survivors_list) > 0:
            # Handle zeros for log scale
            nonzero_data = [x + 0.1 for x in survivors_list if x > 0]
            if len(nonzero_data) > 0:
                bins = np.logspace(np.log10(min(nonzero_data)), np.log10(max(nonzero_data)), 30)
                plt.hist(nonzero_data, bins=bins, alpha=0.7, color='forestgreen')
                plt.xscale('log')
                plt.xlabel("Number of Resistant Survivors (log scale)")
                plt.ylabel("Frequency")
                plt.title("Log-Scale Distribution")
        
        plt.tight_layout()
        save_path = os.path.join(results_path, f"distribution_{model_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        return save_path
    except ImportError as e:
        logging.warning(f"Visualization libraries not available: {e}")
        return None

def create_comparison_report(models_data, results_path):
    """Create a text report comparing the statistics of different models."""
    report_path = os.path.join(results_path, "model_comparison_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Luria-Delbrück Experiment: Model Comparison Report\n")
        f.write("="*80 + "\n\n")
        
        # Write introduction
        f.write("This report compares different models of mutation in the Luria-Delbrück experiment:\n")
        f.write("1. Random (Darwinian) Model: Mutations occur spontaneously during growth\n")
        f.write("2. Induced (Lamarckian) Model: Mutations occur only in response to selective pressure\n")
        if "combined" in models_data:
            f.write("3. Combined Model: Both mechanisms operate simultaneously\n")
        f.write("\n")
        
        # Write detailed statistics table
        f.write("-"*80 + "\n")
        f.write("Detailed Statistics:\n")
        f.write("-"*80 + "\n\n")
        
        # Create table header
        headers = ["Statistic"] + [model.capitalize() for model in models_data.keys()]
        header_line = " ".join(f"{h:<15}" for h in headers)
        f.write(f"{header_line}\n")
        f.write("-" * (15 * len(headers)) + "\n")
        
        # Add rows for each statistic
        stats_to_include = [
            ('mean', 'Mean Survivors'),
            ('variance', 'Variance'),
            ('std_dev', 'Std Deviation'),
            ('median', 'Median'),
            ('variance_to_mean_ratio', 'VMR'),
            ('coefficient_of_variation', 'CV'),
            ('zero_resistant_percent', 'Zero Resistant %')
        ]
        
        for key, label in stats_to_include:
            row = [f"{label:<15}"]
            for model in models_data.keys():
                value = models_data[model][key]
                if isinstance(value, float):
                    row.append(f"{value:>15.2f}")
                else:
                    row.append(f"{value:>15}")
            f.write(" ".join(row) + "\n")
        
        f.write("\n")
        
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
        
        # Compare random vs induced if both exist in the data
        if "random" in models_data and "induced" in models_data:
            random_vmr = models_data["random"]["variance_to_mean_ratio"]
            induced_vmr = models_data["induced"]["variance_to_mean_ratio"]
            
            if random_vmr > induced_vmr:
                f.write("The simulation results support the historical findings: the random model shows significantly\n")
                f.write(f"higher variance-to-mean ratio ({random_vmr:.2f}) compared to the induced model ({induced_vmr:.2f}).\n")
                f.write("This supports the Luria-Delbrück hypothesis that mutations occur spontaneously rather than\n")
                f.write("as a response to selection pressure.\n\n")
            else:
                f.write(f"NOTE: The random model VMR ({random_vmr:.2f}) is not greater than the induced model VMR ({induced_vmr:.2f}).\n")
                f.write("The simulation parameters may need adjustment to better demonstrate the historical findings.\n\n")
    
    logging.info(f"Comparison report created at {report_path}")
    return report_path

# Main function
def main():
    parser = argparse.ArgumentParser(description="Luria-Delbrück Simulation: Random vs. Induced vs. Combined Mutations")
    parser.add_argument("--model", choices=["random", "induced", "combined", "all"], default="random", 
                       help="Which model to simulate? Use 'all' to run all models and compare")
    parser.add_argument("--num_cultures", type=int, default=1000, 
                       help="Number of replicate cultures")
    parser.add_argument("--initial_population_size", type=int, default=10, 
                       help="Initial population size per culture")
    parser.add_argument("--num_generations", type=int, default=5, 
                       help="Number of generations (cell divisions)")
    parser.add_argument("--mutation_rate", type=float, default=0.1, 
                       help="Mutation rate for random/induced/both models")
    parser.add_argument("--induced_mutation_rate", type=float, default=0.1, 
                       help="Induced mutation rate for the combined model")
    parser.add_argument("--n_genes", type=int, default=1, 
                       help="Number of genes in each organism's genome")
    parser.add_argument("--results_path", type=str, default="results", 
                       help="Path to save results")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                       default="INFO", help="Logging level")
    parser.add_argument("--use_parallel", action="store_true", 
                       help="Use parallel processing for simulations")
    parser.add_argument("--n_processes", type=int, default=None, 
                       help="Number of processes for parallel execution")
    parser.add_argument("--optimal_ld", action="store_true", 
                       help="Use optimal parameters for Luria-Delbrück demonstration")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # If optimal_ld flag is set, override parameters with values optimized for demonstrating Luria-Delbrück effect
    if args.optimal_ld:
        args.num_cultures = 2000
        args.initial_population_size = 50
        args.num_generations = 15
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

    # Create results directory if it doesn't exist
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_path, f"simulation_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Initialize logging
    logger, results_dir = init_log(results_dir, args.log_level)

    # Log simulation parameters
    logging.info(f"Starting simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Number of cultures: {args.num_cultures}")
    logging.info(f"Initial population size: {args.initial_population_size}")
    logging.info(f"Number of generations: {args.num_generations}")
    logging.info(f"Mutation rate: {args.mutation_rate}")
    if args.model in ["combined", "all"]:
        logging.info(f"Induced mutation rate: {args.induced_mutation_rate}")
    
    # Set up the Organism class parameters
    Organism.N_GENES = args.n_genes  
    Organism.P_MUTATION = args.mutation_rate
    
    # Dictionary to store results and statistics
    all_results = {}
    all_stats = {}
    
    # Run the requested model(s)
    if args.model == "all":
        models_to_run = ["random", "induced", "combined"]
    else:
        models_to_run = [args.model]
    
    for model in models_to_run:
        logging.info(f"\n{'='*50}")
        logging.info(f"Running {model.capitalize()} Mutation Model")
        logging.info(f"{'='*50}")
        
        if model == "combined":
            survivors_list, sim_time = simulate_model(
                model, args.num_cultures, args.initial_population_size, args.num_generations,
                args.mutation_rate, args.induced_mutation_rate, args.use_parallel, args.n_processes
            )
        else:
            survivors_list, sim_time = simulate_model(
                model, args.num_cultures, args.initial_population_size, args.num_generations,
                args.mutation_rate, None, args.use_parallel, args.n_processes
            )
        
        # Store results
        all_results[model] = survivors_list
        
        # Analyze results
        stats, stats_time = analyze_results(survivors_list, model)
        all_stats[model] = stats
        
        # Log completion and stats
        logging.info(f"Simulation completed in {sim_time:.2f} seconds")
        logging.info(f"Analysis completed in {stats_time:.2f} seconds")
        
        for key, value in stats.items():
            if isinstance(value, float):
                logging.info(f"{key}: {value:.4f}")
            else:
                logging.info(f"{key}: {value}")
        
        # Create visualization
        vis_path = create_visualization(survivors_list, model, results_dir)
        if vis_path:
            logging.info(f"Visualization saved to: {vis_path}")
        
        # Save raw data
        data_path = os.path.join(results_dir, f"survivors_{model}.csv")
        np.savetxt(data_path, survivors_list, fmt='%d', delimiter=',')
        logging.info(f"Raw data saved to: {data_path}")
        
        # Print summary to console
        print(f"\n{model.capitalize()} Model Results:")
        print(f"Mean survivors: {stats['mean']:.2f}")
        print(f"Variance: {stats['variance']:.2f}")
        print(f"Variance-to-Mean Ratio: {stats['variance_to_mean_ratio']:.2f}")
        print(f"Simulation time: {sim_time:.2f} seconds")
    
    # Create comparison report if multiple models were run
    if len(all_stats) > 1:
        compare_path = create_comparison_report(all_stats, results_dir)
        logging.info(f"Comparison report created at: {compare_path}")
        
        # Print key comparison to console
        print("\nModel Comparison (Variance-to-Mean Ratio):")
        for model, stats in all_stats.items():
            print(f"{model.capitalize()}: {stats['variance_to_mean_ratio']:.2f}")
        
        if "random" in all_stats and "induced" in all_stats:
            random_vmr = all_stats["random"]["variance_to_mean_ratio"]
            induced_vmr = all_stats["induced"]["variance_to_mean_ratio"]
            print(f"\nRandom/Induced VMR ratio: {random_vmr/induced_vmr:.2f}")
            
            if random_vmr > induced_vmr:
                print("✓ Results support the Luria-Delbrück hypothesis")
            else:
                print("✗ Results do not clearly support the Luria-Delbrück hypothesis")
    
    # Record and log total execution time
    total_time = time.time() - start_time
    logging.info(f"Total execution time: {total_time:.2f} seconds")
    print(f"\nAll results saved to: {results_dir}")
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()



# python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/src/Simmain.py  --model all --num_cultures 1000 --initial_population_size 10 --num_generations 5 --mutation_rate 0.01 --induced_mutation_rate 0.01 --results_path results --use_parallel