(sexy_yeast_env) 07:47:10 🖤 barc@cn769:~ > py /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/main.py --model random --num_cultures 1 --initial_population_size 10 --num_generations 2 --mutation_rate 0.01 --Results_path /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/Results --log_level INFO  --optimize_numpy --help
usage: main.py [-h] [--model {random,induced,combined,all}] [--num_cultures NUM_CULTURES]
               [--initial_population_size INITIAL_POPULATION_SIZE] [--num_generations NUM_GENERATIONS]
               [--mutation_rate MUTATION_RATE] [--induced_mutation_rate INDUCED_MUTATION_RATE]
               [--n_genes N_GENES] [--Results_path RESULTS_PATH]
               [--log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--use_parallel]
               [--n_processes N_PROCESSES] [--profile] [--optimize_numpy]

Luria–Delbrück Simulation: Random vs. Induced vs. Combined

options:
  -h, --help            show this help message and exit
  --model {random,induced,combined,all}
                        Which model to simulate? Use 'all' to run all models and compare
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
  --n_genes N_GENES     Number of genes in each organism's genome (first gene determines resistance)
  --Results_path RESULTS_PATH
                        Path to the Results directory
  --log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level
  --use_parallel        Use parallel processing for simulations
  --n_processes N_PROCESSES
                        Number of processes to use (default: CPU count - 1)
  --profile             Run with profiling to identify bottlenecks
  --optimize_numpy      Use numpy vectorization where possible




py /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/main.py --model random --num_cultures 6 --initial_population_size 1000 --num_generations 4 --mutation_rate 0.01 --Results_path /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/Results --log_level INFO



bsub -q short -R rusage[mem=10GB] py /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/main.py --model random --num_cultures 40 --initial_population_size 1000 --num_generations 10 --mutation_rate 0.01 --Results_path /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/Results --log_level INFO 




bsub -q short -R rusage[mem=10GB] python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/main.py --model random --num_cultures 64 --initial_population_size 1000 --num_generations 10 --mutation_rate 0.01 --Results_path /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/Results --log_level INFO --use_parallel --optimize_numpy --n_processes 10 --n_genes 1
bsub -q short -R rusage[mem=10GB] python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/main.py --model random --num_cultures 64 --initial_population_size 1000 --num_generations 100 --mutation_rate 0.01 --Results_path /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/Results --log_level INFO --use_parallel --optimize_numpy --n_processes 10 --n_genes 1
bsub -q short -R rusage[mem=10GB] python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/main.py --model all --num_cultures    64 --initial_population_size 1000 --num_generations 10 --mutation_rate 0.01 --Results_path /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/Results --log_level INFO --use_parallel --optimize_numpy --n_processes 10 --n_genes 1
bsub -q short -R rusage[mem=10GB] python /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/main.py --model all --num_cultures    64 --initial_population_size 1000 --num_generations 100 --mutation_rate 0.01 --Results_path /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Assignment_Introduction_Luria_Delbrook/Results --log_level INFO --use_parallel --optimize_numpy --n_processes 10 --n_genes 1