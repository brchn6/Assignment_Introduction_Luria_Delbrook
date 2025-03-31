Assignment_Introduction_Luria_Delbrook/main.py --n_generations 4 --n_organisms 6 --mutation_rate 0.01 --genome_length 32 --help
usage: main.py [-h] [--model {random,induced,combined}] [--num_cultures NUM_CULTURES]
               [--initial_population_size INITIAL_POPULATION_SIZE] [--num_generations NUM_GENERATIONS]
               [--mutation_rate MUTATION_RATE] [--induced_mutation_rate INDUCED_MUTATION_RATE]
               [--Results_path RESULTS_PATH] [--log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Luria–Delbrück Simulation: Random vs. Induced vs. Combined

options:
  -h, --help            show this help message and exit
  --model {random,induced,combined}
                        Which model to simulate?
  --num_cultures NUM_CULTURES
                        Number of replicate cultures
  --initial_population_size INITIAL_POPULATION_SIZE
                        Initial population size per culture
  --num_generations NUM_GENERATIONS
                        Number of generations (cell divisions)
  --mutation_rate MUTATION_RATE
                        Mutation rate (used as either random, induced, or both)
  --induced_mutation_rate INDUCED_MUTATION_RATE
                        Separate induced mutation rate for the combined model
  --Results_path RESULTS_PATH
                        Path to the Resu directory
  --log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level



py /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/main.py --model random --num_cultures 6 --initial_population_size 1000 --num_generations 4 --mutation_rate 0.01 --Results_path /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/Results --log_level INFO