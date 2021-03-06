PX_OUTPUT_FOLDER = './results/px_results'
RANDOM_SEED = 7101856  # Nikola Tesla's Birthday
SAT_PROBLEMS_FOLDER = './sat_problems/trial_problems'
TRIALS_PER_PROBLEM = 10
UBC_OUTPUT_FOLDER = './results/ubcsat_outputs'
UBC_SAT_PATH = './sat_solvers/ubcsat.exe'


def print_verbose(x, value: int, threshold: int):
    if value >= threshold:
        print(x)
