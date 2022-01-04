from constants import RANDOM_SEED, SAT_PROBLEMS_FOLDER, TRIALS_PER_PROBLEM, UBC_OUTPUT_FOLDER, UBC_SAT_PATH

import os
import subprocess
from tqdm.auto import tqdm


def run_solver(instance_file, output_file=None, algorithm='g2wsat', runs=2, seed=None, cutoff=None, timeout=None):

    command = f'{UBC_SAT_PATH} -alg {algorithm} -i {instance_file} -r bestsol -r unsatclauses -runs {runs}'

    if seed is not None:
        command += f' -seed {seed}'

    if cutoff is not None:
        command += f' -cutoff {cutoff}'

    if timeout is not None:
        command += f' -timeout {timeout}'

    print(f'\nRunning command \'{command}\'')
    if output_file is None:
        output = subprocess.run(command, capture_output=True).stdout.decode('utf-8')
        return output
    else:
        file = open(output_file, 'w')
        subprocess.run(command, stdout=file)


def main():
    if not os.path.exists(UBC_OUTPUT_FOLDER):
        os.makedirs(UBC_OUTPUT_FOLDER)

    for input_file in tqdm(os.listdir(SAT_PROBLEMS_FOLDER)):

        output_file = f'{input_file[0:-4]}.txt'
        
        input_path = os.path.join(SAT_PROBLEMS_FOLDER, input_file)
        output_path = os.path.join(UBC_OUTPUT_FOLDER, output_file)

        if output_file not in os.listdir(UBC_OUTPUT_FOLDER):
            run_solver(input_path, output_file=output_path, runs=(2 * TRIALS_PER_PROBLEM), seed=RANDOM_SEED)


if __name__ == '__main__':
    main()
