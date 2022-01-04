from sat import SATProblem

import networkx as nx
import numpy as np


def get_vig(sat, verbose=False):
    """
    Takes in a SAT problem and returns its Variable Interaction Graph (VIG)

    Arguments
    sat (SATProblem): The SAT problem to create a VIG for
    verbose (boolean): Whether to print extra information

    Returns
    vig (networkx.classes.graph.Graph): The VIG
    """

    vig = nx.Graph()

    assert (len(sat.clauses) == sat.m)

    # number of clauses
    for i in range(sat.m):

        if len(sat.clauses[i]) == 1:
            vig.add_node(abs(sat.clauses[i][0]))

        else:
            # K-bounded.  For example, in MAX-K-SAT, loops j and k together will only run (K choose 2) times
            for j in range(len(sat.clauses[i])):
                for k in range(j + 1, len(sat.clauses[i])):
                    if verbose:
                        print(abs(sat.clauses[i][j]), abs(sat.clauses[i][k]))
                    vig.add_edge(abs(sat.clauses[i][j]), abs(sat.clauses[i][k]))

    assert (sat.n == vig.number_of_nodes())

    return vig


def decompose_vig(vig, p1, p2):
    """
    Decomposes a VIG using two solutions.
    If a variable has the same assignment in both solutions,
    its node is removed from the graph.

    Arguments
    vig (networkx.classes.graph.Graph): The VIG to decompose. This graph object will remain unchanged
    p1 (array-like): A list of variable assignments
    p2 (array-like): A list of variable assignments

    Returns
    decomposed_graph (networkx.classes.graph.Graph): The decomposed graph
    """

    if len(p1) != len(p2):
        raise ValueError(f'Solutions P1 and P2 must have the same length.  Got lengths {len(p1)} and {len(p2)}')

    decomposed_graph = vig.copy()

    for i in range(len(p1)):
        if p1[i] == p2[i]:
            if decomposed_graph.has_node(i + 1):
                decomposed_graph.remove_node(i + 1)

    return decomposed_graph


def decompose_problem(sat, p1_sat, p2_sat, verbose=False):
    """
    Decomposes the SAT problem by only including clauses with variables that appear in an unsatisfied clause.

    Arguments
    sat (SATProblem): The original problem to be decomposed
    p1_sat (np.ndarray of bool): A numpy array of booleans indicating which clauses P1 has satisfied
    p2_sat (np.ndarray of bool): A numpy array of booleans indicating which clauses P2 has satisfied
    verbose (bool): Whether to print extra information

    Returns
    decomposed_sat (SATProblem): The decomposed SAT problem
    """

    if len(p1_sat) != len(p2_sat):
        raise ValueError(f'P1 and P2 indicate different number of clauses.  '
                         f'Got numbers of clauses {len(p1_sat)} and {len(p2_sat)}')
    if len(p1_sat) != sat.m:
        raise ValueError(f'P1 and P2 indicate a different number of clauses than the SAT problem.  '
                         f'Got numbers of clauses {len(p1_sat)} and {sat.m}')

    # get all the unsat clauses
    unsat_clauses = sat.clauses[~(p1_sat & p2_sat)]
    if verbose:
        print(f'Unsat clauses: {unsat_clauses}')

    # get all the variables in those clauses
    variables = set()
    for clause in unsat_clauses:
        for variable in clause:
            variables.add(abs(variable))
    if verbose:
        print(f'Variables that appear in unsat clauses: {variables}')

    # get all clauses with those variables
    decomposed_sat_clauses = []
    for clause in sat.clauses:
        for variable in clause:
            if abs(variable) in variables:
                decomposed_sat_clauses.append(list(clause))
                break
    if verbose:
        print(f'Clauses that contain those variables: {decomposed_sat_clauses}')

    decomposed_sat = SATProblem(np.array(decomposed_sat_clauses, dtype=object), name=f'decomposed_{sat.name}')

    return decomposed_sat


def partition_crossover(sat, decomposed_vig, p1, p2, verbose=0):
    new_solution = np.array([None] * len(p1))

    # set common variables
    for i in range(len(p1)):
        if p1[i] == p2[i]:
            new_solution[i] = p1[i]

    if verbose >= 2:
        print(f'Common variable assignments: {new_solution}')

    for component in nx.connected_components(decomposed_vig):
        if verbose >= 1:
            print(f'Component: {component}')

        # get the sub_problem of f(x), g(x).  We use g(x) to evaluate P1 and P2 for the component
        sub_sat = get_sub_problem(sat, component)
        if verbose >= 2:
            print(f'\tSub problem clauses: {sub_sat.clauses}')

        p1_sub_solution = [p1[variable - 1] for variable in component]
        p2_sub_solution = [p2[variable - 1] for variable in component]
        if verbose >= 2:
            print(f'\tP1 assignments: {p1_sub_solution}')
            print(f'\tP2 assignments: {p2_sub_solution}')

        p1_score = sub_sat.score_solution(p1_sub_solution)
        p2_score = sub_sat.score_solution(p2_sub_solution)
        if verbose >= 1:
            print(f'\tP1 Score: {p1_score}, P2 Score: {p2_score}')

        if p1_score >= p2_score:
            for variable in component:
                new_solution[variable - 1] = p1[variable - 1]
        else:
            for variable in component:
                new_solution[variable - 1] = p2[variable - 1]

    if verbose >= 1:
        print(f'Solution after recombination: {new_solution}')
        print(f'Filling in None spots with assignments from P1')

    for i in range(len(new_solution)):
        if new_solution[i] is None:
            new_solution[i] = p1[i]

    assert(sat.score_solution(new_solution) >= sat.score_solution(p1))
    assert(sat.score_solution(new_solution) >= sat.score_solution(p2))

    return new_solution


def get_sub_problem(sat, component):

    sub_problem_clauses = []
    for clause in sat.clauses:
        sub_clause = []
        for variable in clause:
            if abs(variable) in component:
                sub_clause.append(variable)
        if len(sub_clause) > 0:
            sub_problem_clauses.append(sub_clause)

    sub_sat = SATProblem(sub_problem_clauses, name=f'sub_{sat.name}')

    return sub_sat
