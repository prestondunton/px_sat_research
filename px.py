from sat import SATProblem, get_unique_vars
from constants import print_verbose

import networkx as nx
import numpy as np


def get_vig(sat, verbose=0):
    """
    Takes in a SAT problem and returns its Variable Interaction Graph (VIG)

    Arguments
    sat (SATProblem): The SAT problem to create a VIG for
    verbose (int): Level of which to print extra information

    Returns
    vig (networkx.classes.graph.Graph): The VIG
    """

    vig = nx.Graph()

    assert (len(sat.clauses) == sat.m)

    # number of clauses
    for i in range(sat.m):

        if len(sat.clauses[i]) == 1:
            vig.add_node(abs(list(sat.clauses[i])[0]))

        else:
            # K-bounded.  For example, in MAX-K-SAT, loops j and k together will only run (K choose 2) times
            for variable_1 in sat.clauses[i]:
                for variable_2 in sat.clauses[i]:
                    if variable_1 < variable_2:
                        print_verbose(f'{abs(variable_1)} {abs(variable_2)}', verbose, 1)
                        vig.add_edge(abs(variable_1), abs(variable_2))

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


def decompose_problem(sat, p1, p2, p1_unsat, p2_unsat, init_method, verbose=0):
    """

    Returns
    decomposed_sat (SATProblem): The decomposed problem
    iterations (int): How many times the while loop ran
    """
    assert(sat.m == len(sat.clauses))

    if len(p1) != len(p2):
        raise ValueError(f'P1 and P2 must be the same lengths.  Got lengths {len(p1)} and {len(p2)}')

    swapc, var = init_swapc_var(sat, p1, p2, p1_unsat, p2_unsat, init_method, verbose)
    swapc, var, iterations = grow_swapc_var(sat, p1, p2, swapc, var, verbose)

    new_clauses = []
    for clause in sat.clauses[list(swapc)]:
        new_clause = set([])
        for variable in clause:
            if abs(variable) in var:
                new_clause.add(variable)
        new_clauses.append(new_clause)

    decomposed_sat = SATProblem(np.array(new_clauses), name=f'decomposed_{sat.name}')
    assert(len(swapc) == decomposed_sat.m)
    assert(var == decomposed_sat.unique_vars)

    return decomposed_sat, iterations


def init_swapc_var(sat, p1, p2, p1_unsat, p2_unsat, method, verbose=0):
    if method not in ['p1', 'p2', 'xor']:
        raise ValueError(f'Invalid SWAPC/VAR initialization method: {method}')

    if len(p1_unsat) == 0:
        raise ValueError('Every clause is satisfied by P1.  No need for preprocessor.')
    if len(p2_unsat) == 0:
        raise ValueError('Every clause is satisfied by P2.  No need for preprocessor.')

    # Step 3 and 4
    if method == 'p1':
        swapc = set(p1_unsat) - set(p2_unsat)
    elif method == 'p2':
        swapc = set(p2_unsat) - set(p1_unsat)
    else:
        swapc = set(p1_unsat).symmetric_difference(set(p2_unsat))

    print_verbose(f'[3,4] Length of Init SWAPC: {len(swapc)}', verbose, 1)
    print_verbose(f'[3,4] Init SWAPC: {swapc}', verbose, 2)

    if len(swapc) == 0:
        raise ValueError('SWAPC is initialized to be empty.  Preprocessor failed.')

    # Step 5
    var = get_unique_vars(sat.clauses[list(swapc)])
    print_verbose(f'[5] Length of Init VAR: {len(var)}', verbose, 1)
    print_verbose(f'[5] Init VAR: {var}', verbose, 2)
    var = remove_common_vars(var, p1, p2)
    print_verbose(f'[5] Length of No Common Variables VAR: {len(var)}', verbose, 1)
    print_verbose(f'[5] No Common Variables VAR: {var}', verbose, 2)

    return swapc, var


def grow_swapc_var(sat, p1, p2, swapc, var, verbose=0):
    previous_var = None
    sat_by_common = set(sat_by_common_variable(sat, p1, p2))
    print_verbose(f'Length of sat_by_common: {len(sat_by_common)}', verbose, 1)
    print_verbose(f'sat_by_common: {sat_by_common}', verbose, 2)

    iterations = 0
    while var != previous_var:
        iterations += 1
        previous_var = var

        # Step 6
        swapc = swapc.union(sat.clauses_with_variables(var))
        print_verbose(f'[6] Length of SWAPC: {len(swapc)}', verbose, 1)
        print_verbose(f'[6] SWAPC: {swapc}', verbose, 2)

        # Step 7
        swapc = swapc - sat_by_common
        print_verbose(f'[7] Length of SWAPC: {len(swapc)}', verbose, 1)
        print_verbose(f'[7] SWAPC: {swapc}', verbose, 2)

        # Step 8
        var = var.union(get_unique_vars(sat.clauses[list(swapc)]))
        var = remove_common_vars(var, p1, p2)
        print_verbose(f'[8] Length of VAR: {len(var)}', verbose, 1)
        print_verbose(f'[8] VAR: {var}', verbose, 2)

    print_verbose(f'Loop ran {iterations} times', verbose, 1)

    return swapc, var, iterations


def remove_common_vars(var, p1, p2):
    if len(p1) != len(p2):
        raise ValueError(f'P1 and P2 must be the same lengths.  Got lengths {len(p1)} and {len(p2)}')

    for i in range(len(p1)):
        if abs(p1[i-1]) in var:
            if p1[i-1] == p2[i-1]:
                var.remove(abs(p1[i-1]))

    return var


def sat_by_common_variable(sat, p1, p2):
    if len(p1) != len(p2):
        raise ValueError(f'P1 and P2 must be the same lengths.  Got lengths {len(p1)} and {len(p2)}')

    p1_satisfying_vars = sat.sat_by_variable(p1)
    p2_satisfying_vars = sat.sat_by_variable(p2)

    sat_by_common = [len(p1_satisfying_vars[i] & p2_satisfying_vars[i]) > 0 for i in range(len(p1_satisfying_vars))]

    return np.argwhere(sat_by_common).flatten().tolist()


def partition_crossover(sat, decomposed_vig, p1, p2, none_fill, verbose=0):
    if none_fill not in ['p1', 'p2']:
        raise ValueError(f'Invalid method for filling None assignments: {none_fill}')

    if len(p1) != len(p2):
        raise ValueError(f'P1 and P2 must be the same lengths.  Got lengths {len(p1)} and {len(p2)}')

    new_solution = np.array([None] * len(p1))

    # set common variables
    for i in range(len(p1)):
        if p1[i] == p2[i]:
            new_solution[i] = p1[i]

    print_verbose(f'Common variable assignments: {new_solution}', verbose, 2)

    for component in nx.connected_components(decomposed_vig):
        print_verbose(f'Component: {component}', verbose, 1)

        # get the sub_problem of f(x), g(x).  We use g(x) to evaluate P1 and P2 for the component
        sub_sat = SATProblem(sat.clauses[list(sat.clauses_with_variables(component))], name=f'sub_{sat.name}')
        print_verbose(f'\tSub problem clauses: {sub_sat.clauses}', verbose, 2)

        p1_score = sub_sat.score_solution(p1)
        p2_score = sub_sat.score_solution(p2)
        print_verbose(f'\tP1 Score: {p1_score}, P2 Score: {p2_score}', verbose, 1)

        if p1_score >= p2_score:
            for variable in component:
                new_solution[variable - 1] = p1[variable - 1]
        else:
            for variable in component:
                new_solution[variable - 1] = p2[variable - 1]

    print_verbose(f'Solution after recombination: {new_solution}', verbose, 1)

    print_verbose(f'Filling in None spots with assignments from {none_fill.upper()}', verbose, 1)
    for i in range(len(new_solution)):
        if new_solution[i] is None:
            if none_fill == 'p1':
                new_solution[i] = p1[i]
            else:
                new_solution[i] = p2[i]

    return new_solution
