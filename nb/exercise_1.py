# %%
import numpy as np

# %%
class Organism:

    # change these required
    P_ACQUIRE_MUTATION = 0.05
    P_LOSE_MUTATION = 0.0

    def __init__(self, gene: bool = False):
        self.gene = gene

    def replicate_gene(self):

        # acquire mutation
        if (not self.gene) and (np.random.rand() < self.P_ACQUIRE_MUTATION):
            copied_gene = True
        # lose mutation (an extension; usually don't happen)
        elif self.gene and (np.random.rand() < self.P_LOSE_MUTATION):
            copied_gene = False
        # standard copy
        else:
            copied_gene = self.gene

        return copied_gene

    def divide(self):
        # Perform binary fission by replicating the genes and dividing into two organisms
        offsprings = [
            Organism(gene=self.replicate_gene()),
            Organism(gene=self.replicate_gene()),
        ]
        return offsprings

# %%
class Experiment:

    # change if required
    N_GENERATIONS = 12

    def __init__(self):
        self.population = []
        self.population_genome = []
        self.mutation_part = []

    def run_generations(self):

        # initialize
        self.population = [[Organism()]]
        self.population_genome.append([x.gene for x in self.population[-1]])
        self.mutation_part.append(np.mean(self.population_genome[-1]))

        for i in range(self.N_GENERATIONS - 1):
            current_generation = self.population[-1]
            new_generation = [offspring for x in current_generation for offspring in x.divide()]
            self.population.append(new_generation)
            self.population_genome.append([x.gene for x in new_generation])
            self.mutation_part.append(np.mean(self.population_genome[-1]))

    def print_population(self):
        for i in range(self.N_GENERATIONS):
            print(f"generation #{i}: {self.mutation_part[i]}")

# %%
experiment = Experiment()

experiment.run_generations()

experiment.print_population()

# %%
def run_experiment_series(n_experiments: int):

    n_generations = Experiment.N_GENERATIONS
    mutation_part_array = np.zeros((n_experiments, n_generations))

    for n in range(n_experiments):
        if n % 10 == 0:
            print(f"experiment #{n}")
        experiment = Experiment()
        experiment.run_generations()
        mutation_part_array[n, :] = np.array(experiment.mutation_part)

    return mutation_part_array

run_experiment_series(n_experiments=10)

# %%
from tqdm import tqdm
import numpy as np

def run_experiment_series(n_experiments: int):
    n_generations = Experiment.N_GENERATIONS
    mutation_part_array = np.zeros((n_experiments, n_generations))

    for n in tqdm(range(n_experiments), desc="Running Experiments"):
        experiment = Experiment()
        experiment.run_generations()
        mutation_part_array[n, :] = np.array(experiment.mutation_part)

    return mutation_part_array

run_experiment_series(n_experiments=500)

# %%
# Plot the histogram
plt.hist(mutation_part_last_generation, bins=15, edgecolor='black', alpha=0.7)
plt.xlabel("Mutation Part Value")
plt.ylabel("Frequency")
plt.title("Histogram of Mutation Part in Last Generation")
plt.xlim([0, 1])
plt.show()

# %%
# --------------------------------------------------

# %%


# %%


# %%


# %%



