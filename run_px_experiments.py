from constants import PX_OUTPUT_FOLDER, SAT_PROBLEMS_FOLDER, TRIALS_PER_PROBLEM
from sat import read_sat_problem, bitstring_to_int_array
from px import get_vig, decompose_vig, decompose_problem, partition_crossover, EmptySWAPCError

import networkx as nx
import numpy as np
import os
import pandas as pd
import sys
import re
from tqdm.auto import tqdm


def run_experiment(sat_file, ubc_file):

    results, sat, vig, solutions, unsat_clauses = init_experiment(sat_file, ubc_file)
    n = sat.n
    m = sat.m
    k = sat.k
    e = vig.number_of_edges()
    c = nx.number_connected_components(vig)

    for i in tqdm(range(0, 2 * TRIALS_PER_PROBLEM, 2), leave=False):
        p1, p2, p1_unsat, p2_unsat, p1_score, p2_score = load_parent_solutions(solutions, unsat_clauses, sat, i)

        if p1_score == sat.m:
            print('P1 completely satisfies the problem.')
            results = results.append({'P1_score': p1_score,
                                      'P2_score': p2_score,
                                      'shared_variables': len(set(p1).intersection(set(p2))),
                                      'n': n,
                                      'm': m,
                                      'k': k}, ignore_index=True)
            continue

        print('Decomposing VIG')
        decomposed_vig = decompose_vig(vig, p1, p2)

        print('Performing PX')
        new_solution = partition_crossover(sat, decomposed_vig, p1, p2, none_fill='p1')

        try:
            print('Decomposing Problem')
            decomposed_sat, iterations = decompose_problem(sat, p1, p2, p1_unsat, p2_unsat, init_method='p1')
        except EmptySWAPCError as exp:
            print(exp.message)
            results = results.append({'P1_score': p1_score,
                                      'P2_score': p2_score,
                                      'NS_score': sat.score_solution(new_solution),
                                      'shared_variables': len(set(p1).intersection(set(p2))),

                                      'n': n,
                                      'm': m,
                                      'k': k,
                                      'e': e,
                                      'c': c,
                                      'n_H': decomposed_vig.number_of_nodes(),
                                      'e_H': decomposed_vig.number_of_edges(),
                                      'q': nx.number_connected_components(decomposed_vig),

                                      'iterations': 0,
                                      'n*': 0,
                                      'm*': 0,
                                      'k*': 0,
                                      'e*': 0,
                                      'c*': 0,
                                      'n_H*': 0,
                                      'e_H*': 0,
                                      'q*': 0
                                      }, ignore_index=True)
            continue

        print('Creating VIG Prime')
        vig_prime = get_vig(decomposed_sat)
        print('Decomposing VIG Prime')
        decomposed_vig_prime = decompose_vig(vig_prime, p1, p2)

        print('Performing PX*')
        new_solution_prime = partition_crossover(decomposed_sat, decomposed_vig_prime, p1, p2, none_fill='p1')

        results = results.append({'P1_score': p1_score,
                                  'P2_score': p2_score,
                                  'NS_score': sat.score_solution(new_solution),
                                  'NS_score*': sat.score_solution(new_solution_prime),
                                  'shared_variables': len(set(p1).intersection(set(p2))),

                                  'n': n,
                                  'm': m,
                                  'k': k,
                                  'e': e,
                                  'c': c,
                                  'n_H': decomposed_vig.number_of_nodes(),
                                  'e_H': decomposed_vig.number_of_edges(),
                                  'q': nx.number_connected_components(decomposed_vig),

                                  'iterations': iterations,
                                  'n*': decomposed_sat.n,
                                  'm*': decomposed_sat.m,
                                  'k*': decomposed_sat.k,
                                  'e*': vig_prime.number_of_edges(),
                                  'c*': nx.number_connected_components(vig_prime),
                                  'n_H*': decomposed_vig_prime.number_of_nodes(),
                                  'e_H*': decomposed_vig_prime.number_of_edges(),
                                  'q*': nx.number_connected_components(decomposed_vig_prime)
                                  }, ignore_index=True)

    return results


def init_experiment(sat_file, ubc_file):
    results = pd.DataFrame(columns=['P1_score', 'P2_score', 'NS_score', 'NS_score*', 'shared_variables',
                                    'n', 'm', 'k', 'e', 'c', 'n_H', 'e_H', 'q',
                                    'iterations', 'n*', 'm*', 'k*', 'e*', 'c*', 'n_H*', 'e_H*', 'q*',
                                    ])

    print('Reading SAT Problem')
    sat = read_sat_problem(sat_file)

    print('Creating VIG')
    vig = get_vig(sat)

    print('Parsing Output')
    solutions = parse_output(ubc_file)
    unsat_clauses = [np.argwhere(~(sat.evaluate_solution(solution))).flatten() for solution in solutions]

    return results, sat, vig, solutions, unsat_clauses


def parse_output(output_file):
    """
    Parses output from the UBCSAT executable.

    Arguments
    output_file (str): UBCSAT output file to read from

    Returns
    solutions (boolean np.ndarray of shape (runs, variables)): The variable assignments for each partial solution
    """

    solution_strings = []

    file = open(output_file, 'r')

    for line in file.readlines():

        if re.match('[0-9]+ [01]+ [0-9]+ [01]+$', line):
            solution_strings.append(line.split(' ')[3].replace('\n', ''))

    solutions = np.array([bitstring_to_int_array(bitstring) for bitstring in solution_strings])

    return solutions


def load_parent_solutions(solutions, unsat_clauses, sat, i):
    """
    Loads the parent solutions for a trial.

    Arguments
    solutions (boolean np.ndarray with shape (runs, variables)): The variable assignments for each partial solution
    unsat_clauses (list of list of int with length (# of solutions)): The indices into sat.clauses of
                                                                       unsatisfied clauses for each solution
    sat (SATProblem): The sat problem to score solutions against
    i (int): The index into solutions and unsat_clauses for the first parent solution

    """

    p1, p2 = solutions[i], solutions[i + 1]
    p1_unsat, p2_unsat = unsat_clauses[i], unsat_clauses[i + 1]
    p1_score = sat.score_solution(p1)
    p2_score = sat.score_solution(p2)

    if p2_score > p1_score:
        p1, p2 = p2, p1
        p1_unsat, p2_unsat = p2_unsat, p1_unsat
        p1_score, p2_score = p2_score, p1_score

    return p1, p2, p1_unsat, p2_unsat, p1_score, p2_score


def main():
    if not os.path.exists(PX_OUTPUT_FOLDER):
        os.makedirs(PX_OUTPUT_FOLDER)

    completed_results = os.listdir(PX_OUTPUT_FOLDER)

    for file in tqdm(os.listdir(sys.argv[1]), leave=False):

        problem_name = file.replace('.txt', '')
        sat_path = os.path.join(SAT_PROBLEMS_FOLDER, problem_name + '.cnf')
        ubc_path = os.path.join(sys.argv[1], problem_name + '.txt')
        csv_file = problem_name + '.csv'
        csv_path = os.path.join(PX_OUTPUT_FOLDER, csv_file)

        if csv_file not in completed_results:
            print(f'Running experiment on {sat_path}')
            results = run_experiment(sat_path, ubc_path)
            print(results)
            results.to_csv(csv_path)


if __name__ == '__main__':
    main()
