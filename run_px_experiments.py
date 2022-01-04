from constants import PX_OUTPUT_FOLDER, SAT_PROBLEMS_FOLDER, TRIALS_PER_PROBLEM, UBC_OUTPUT_FOLDER
from sat import read_sat_problem
from px import get_vig, decompose_vig, decompose_problem, partition_crossover

import networkx as nx
import numpy as np
import os
import pandas as pd
import re
from tqdm.auto import tqdm


def run_experiment(sat_file, ubc_file):
    results = pd.DataFrame(columns=['P1 Score', 'P2 Score', 'Both Score',
                                    'n', 'm', 'k', 'e', 'c',
                                    'n_H', 'e_H', 'q',
                                    'n\'', 'm\'', 'k\'', 'e\'', 'c\'',
                                    'n_H\'', 'e_H\'', 'q\''
                                    ])

    print('Reading SAT Problem')
    sat = read_sat_problem(sat_file)

    print('Creating VIG')
    vig = get_vig(sat)

    n = vig.number_of_nodes()
    m = sat.m
    e = vig.number_of_edges()
    c = nx.number_connected_components(vig)
    k = sat.k

    print('Parsing Output')
    solutions, sat_clauses = parse_output(ubc_file)

    for i in tqdm(range(0, 2 * TRIALS_PER_PROBLEM, 2), leave=False):
        p1, p2 = solutions[i], solutions[i + 1]
        p1_sat, p2_sat = sat_clauses[i], sat_clauses[i + 1]

        print('Decomposing VIG')
        decomposed_vig = decompose_vig(vig, p1, p2)

        print('Decomposing Problem')
        decomposed_sat = decompose_problem(sat, p1_sat, p2_sat)
        print('Creating VIG Prime')
        vig_prime = get_vig(decomposed_sat)
        print('Decomposing VIG Prime')
        decomposed_vig_prime = decompose_vig(vig_prime, p1, p2)

        # print('Performing PX')
        # new_solution = partition_crossover(sat, decomposed_vig, p1, p2, verbose=1)
        # print('Performing PX\'')
        # new_solution_prime = partition_crossover(sat, decomposed_vig_prime, p1, p2, verbose=1)

        results = results.append({'P1 Score': sum(p1_sat),
                                  'P2 Score': sum(p2_sat),
                                  'Both Score': sum(p1_sat & p2_sat),
                                  #                         'PX Score': sat.score_solution(new_solution),
                                  #                         'PX\' Score': sat.score_solution(new_solution_prime),

                                  'n': n,
                                  'm': m,
                                  'k': k,
                                  'e': e,
                                  'c': c,
                                  'n_H': decomposed_vig.number_of_nodes(),
                                  'e_H': decomposed_vig.number_of_edges(),
                                  'q': nx.number_connected_components(decomposed_vig),

                                  'n\'': vig_prime.number_of_nodes(),
                                  'm\'': decomposed_sat.m,
                                  'k\'': decomposed_sat.k,
                                  'e\'': vig_prime.number_of_edges(),
                                  'c\'': nx.number_connected_components(vig_prime),
                                  'n_H\'': decomposed_vig_prime.number_of_nodes(),
                                  'e_H\'': decomposed_vig_prime.number_of_edges(),
                                  'q\'': nx.number_connected_components(decomposed_vig_prime)
                                  }, ignore_index=True)

    return results


def parse_output(output_file):
    """
    Parses output from the UBCSAT executable.

    Arguments
    output_file (str): UBCSAT output file to read from

    Returns
    solutions (boolean np.ndarray of shape (runs, variables)): The variable assignments for each partial solution
    sat_clauses (boolean np.ndarray of shape (runs, clauses)): The satisfied clauses for each partial solution
    """

    solution_strings = []
    sat_clauses_strings = []

    file = open(output_file, 'r')

    for line in file.readlines():

        if re.match('[0-9]+ [01]+ [0-9]+ [01]+$', line):
            solution_strings.append(line.split(' ')[3].replace('\n', ''))

        if re.match('[0-9]+ [01]+$', line):
            sat_clauses_strings.append(line.split(' ')[1].replace('\n', ''))

    solutions = np.array([[bool(int(bit)) for bit in bitstring] for bitstring in solution_strings])
    sat_clauses = np.array([[bool(int(bit)) for bit in bitstring] for bitstring in sat_clauses_strings])

    return solutions, sat_clauses


def format_results(results, ratios=False, mean_std=False):
    results['P1 %SAT'] = results['P1 Score'] / results['m']
    results['P2 %SAT'] = results['P2 Score'] / results['m']
    results['Both %SAT'] = results['Both Score'] / results['m']
    # results['PX %SAT'] = results['PX Score'] / results['m']
    # results['PX\' %SAT'] = results['PX\' Score'] / results['m']

    # results['Decomp Rate'] = results['q'] / results['c']
    # results['Decomp Rate\''] = results['q\''] / results['c\'']

    if ratios:
        results['n / n\''] = results['n'] / results['n\'']
        results['m / m\''] = results['m'] / results['m\'']
        results['k / k\''] = results['k'] / results['k\'']
        results['e / e\''] = results['e'] / results['e\'']
        results['c / c\''] = results['c'] / results['c\'']
        results['n_H / n_H\''] = results['n_H'] / results['n_H\'']
        results['e_H / e_H\''] = results['e_H'] / results['e_H\'']
        results['q / q\''] = results['q'] / results['q\'']
    #   results['Decomp Rate / Decomp Rate\''] = results['Decomp Rate'] / results['Decomp Rate\'']

    if mean_std:
        results.loc['mean'] = results.mean()
        results.loc['std'] = results.std()

    return results


def main():
    if not os.path.exists(PX_OUTPUT_FOLDER):
        os.makedirs(PX_OUTPUT_FOLDER)

    completed_results = os.listdir(PX_OUTPUT_FOLDER)

    for file in tqdm(os.listdir(UBC_OUTPUT_FOLDER), leave=False):

        problem_name = file.replace('.txt', '')
        sat_file = os.path.join(SAT_PROBLEMS_FOLDER, problem_name + '.cnf')
        ubc_file = os.path.join(UBC_OUTPUT_FOLDER, problem_name + '.txt')
        csv_file = os.path.join(PX_OUTPUT_FOLDER, problem_name + '.csv')

        if csv_file not in completed_results:
            result = run_experiment(sat_file, ubc_file)
            result = format_results(result, ratios=True)
            result.to_csv(csv_file)


if __name__ == '__main__':
    main()
