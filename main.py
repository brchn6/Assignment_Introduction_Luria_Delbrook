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
    N_GENES = 10  # Default, overridden in main
    P_MUTATION = 0.1  # Default, overridden in main

    def __init__(self, genome=None, id=None):
        if genome is not None:
            self.genome = genome
        else:
            self.genome = np.random.randint(0, 2, Organism.N_GENES)
        
        self.id = id if id is not None else uuid.uuid4()

    def __repr__(self):
        return f"Organism {self.id} with genome {self.genome}"

    def mutate(self):
        for i in range(Organism.N_GENES):
            if np.random.rand() < Organism.P_MUTATION:
                self.genome[i] = 1 - self.genome[i]  # Flip bit

    def reproduce(self):
        offspring = Organism(genome=self.genome.copy())
        offspring.mutate()
        return offspring


#######################################################################
# Simulation_execution_functions
#######################################################################
def simulate_random_mutations(num_cultures=1000, initial_population_size=10, num_generations=5, mutation_rate=0.1):
    """
    Random (Darwinian) mutation model.
    Mutations arise spontaneously at each generation.
    Returns a list of 'survivors' (resistant cell counts) across cultures.
    """
    survivors_list = []

    for _ in range(num_cultures):
        # Initialize population. All start as sensitive (resistance=0).
        # Each cell is just an integer 0 or 1 representing the 'resistance gene'.
        population = np.zeros(initial_population_size, dtype=int)

        # Simulate G generations.
        for _gen in range(num_generations):
            new_population = []
            for cell in population:
                # Each cell reproduces into two offspring.
                # Offspring 1:
                offspring1 = cell
                if np.random.rand() < mutation_rate:
                    offspring1 = 1 - offspring1  # flip 0->1 or 1->0

                # Offspring 2:
                offspring2 = cell
                if np.random.rand() < mutation_rate:
                    offspring2 = 1 - offspring2

                new_population.append(offspring1)
                new_population.append(offspring2)

            population = np.array(new_population, dtype=int)

        # After final generation, apply selection:
        # survivors are those with the resistance bit = 1
        survivors = np.sum(population == 1)
        survivors_list.append(survivors)

    return survivors_list

def simulate_induced_mutations(num_cultures=1000, initial_population_size=10, num_generations=5, mutation_rate=0.1):
    """
    Induced (Lamarckian) mutation model.
    No mutations during growth; only at the end (when selective agent is applied).
    Returns a list of 'survivors' (resistant cell counts) across cultures.
    """
    survivors_list = []

    for _ in range(num_cultures):
        # Initialize population, all sensitive
        population = np.zeros(initial_population_size, dtype=int)

        # Grow for G generations with no mutations
        for _gen in range(num_generations):
            population = np.repeat(population, 2)  # each cell divides into two identical copies

        # Now apply selection, but in a Lamarckian sense:
        # "Mutations" (to resistance=1) only happen *after* growth:
        # Suppose each cell has a probability `mutation_rate` to become resistant right at the end:
        for i in range(len(population)):
            if np.random.rand() < mutation_rate:
                population[i] = 1

        # Count survivors
        survivors = np.sum(population == 1)
        survivors_list.append(survivors)

    return survivors_list

def simulate_combined_mutations(num_cultures=1000,initial_population_size=10,num_generations=5,random_mut_rate=0.1,induced_mut_rate=0.1):
    """
    Combined model:
    - Random mutations occur during each generation.
    - Additional induced mutations occur at the end.
    """
    survivors_list = []

    for _ in range(num_cultures):
        # Initialize population, all sensitive
        population = np.zeros(initial_population_size, dtype=int)

        # Growth with random mutations
        for _gen in range(num_generations):
            new_population = []
            for cell in population:
                offspring1 = cell
                if np.random.rand() < random_mut_rate:
                    offspring1 = 1 - offspring1
                offspring2 = cell
                if np.random.rand() < random_mut_rate:
                    offspring2 = 1 - offspring2

                new_population.append(offspring1)
                new_population.append(offspring2)
            population = np.array(new_population, dtype=int)

        # Lamarckian-like induced mutation at the end
        for i in range(len(population)):
            if np.random.rand() < induced_mut_rate:
                population[i] = 1

        # Count survivors
        survivors = np.sum(population == 1)
        survivors_list.append(survivors)

    return survivors_list

#######################################################################
# Main
#######################################################################
def main():
    parser = argparse.ArgumentParser(description="Luria–Delbrück Simulation: Random vs. Induced vs. Combined")
    parser.add_argument("--model", choices=["random", "induced", "combined"], default="random", help="Which model to simulate?")
    parser.add_argument("--num_cultures", type=int, default=1000, help="Number of replicate cultures")
    parser.add_argument("--initial_population_size", type=int, default=10, help="Initial population size per culture")
    parser.add_argument("--num_generations", type=int, default=5, help="Number of generations (cell divisions)")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate (used as either random, induced, or both)")
    parser.add_argument("--induced_mutation_rate", type=float, default=0.1, help="Separate induced mutation rate for the combined model")
    parser.add_argument("--Results_path", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Path to the Resu directory")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Logging level")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # set the Results_path to the current directory if not exist make it
    if not os.path.exists(args.Results_path):
        os.makedirs(args.Results_path)
        
    # Initialize logging
    logger = init_log(args.Results_path, args.log_level)
    
    # Log all simulation parameters
    logger.info(f"Starting simulation with parameters:")
    logger.info(f"Model: {args.model}")
    logger.info(f"Number of cultures: {args.num_cultures}")
    logger.info(f"Initial population size: {args.initial_population_size}")
    logger.info(f"Number of generations: {args.num_generations}")
    logger.info(f"Mutation rate: {args.mutation_rate}")
    if args.model == "combined":
        logger.info(f"Induced mutation rate: {args.induced_mutation_rate}")
    
    # Run the appropriate simulation model
    logger.info(f"Running {args.model} mutation model simulation...")
    
    if args.model == "random":
        survivors_list = simulate_random_mutations(
            num_cultures=args.num_cultures,
            initial_population_size=args.initial_population_size,
            num_generations=args.num_generations,
            mutation_rate=args.mutation_rate
        )
    elif args.model == "induced":
        survivors_list = simulate_induced_mutations(
            num_cultures=args.num_cultures,
            initial_population_size=args.initial_population_size,
            num_generations=args.num_generations,
            mutation_rate=args.mutation_rate
        )
    else:  # "combined"
        survivors_list = simulate_combined_mutations(
            num_cultures=args.num_cultures,
            initial_population_size=args.initial_population_size,
            num_generations=args.num_generations,
            random_mut_rate=args.mutation_rate,
            induced_mut_rate=args.induced_mutation_rate
        )
    
    # Log completion of simulation
    logger.info(f"Simulation completed successfully.")

    # Compute statistics
    mean_survivors = np.mean(survivors_list)
    var_survivors = np.var(survivors_list)
    
    # Log the statistics
    logger.info(f"Mean # survivors: {mean_survivors:.3f}")
    logger.info(f"Variance of # survivors: {var_survivors:.3f}")
    
    # Also print to console
    print(f"Model: {args.model}")
    print(f"Number of cultures: {args.num_cultures}")
    print(f"Mean # survivors: {mean_survivors:.3f}")
    print(f"Variance of # survivors: {var_survivors:.3f}")

    # Log histogram creation attempt
    try:
        import matplotlib.pyplot as plt
        
        logger.info("Creating histogram of results...")
        
        plt.hist(survivors_list, bins=range(0, max(survivors_list)+2))
        plt.title(f"Distribution of Survivors ({args.model.capitalize()} Model)")
        plt.xlabel("Number of Resistant Survivors")
        plt.ylabel("Count of Cultures")
        
        # Save the figure to the results directory
        histogram_path = os.path.join(args.Results_path, f"histogram_{args.model}.png")
        plt.savefig(histogram_path)
        logger.info(f"Histogram saved to {histogram_path}")
        
        plt.show()
    except ImportError:
        logger.warning("matplotlib not installed, skipping histogram.")
        print("matplotlib not installed, skipping histogram.")
    except Exception as e:
        logger.error(f"Error creating histogram: {str(e)}")

    # Log the location of the log file
    logger.info(f"Log file saved to {os.path.join(args.Results_path, 'Assignment_Introduction_Luria_Delbrook.log')}")





    
    logger.info("Analysis complete.")


#######################################################################
# Entry point
#######################################################################
if __name__ == "__main__":
    main()


#######################################################################
# Example usage
#######################################################################
# python main.py --model random --num_cultures 1000 --initial_population_size 10 --num_generations 5 --mutation_rate 0.01