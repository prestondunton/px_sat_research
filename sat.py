import lzma
import numpy as np
import re


def get_unique_vars(clauses):
    unique_vars = set()
    for i in range(len(clauses)):
        for j in range(len(clauses[i])):
            unique_vars.add(abs(clauses[i][j]))

    return unique_vars


class SATProblem:
    def __init__(self, clauses, name=None):
        self.name = name
        self.clauses = clauses
        self.unique_vars = get_unique_vars(clauses)

        self.n = len(self.unique_vars)
        self.m = len(clauses)
        self.k = max([len(clause) for clause in clauses])

    def evaluate_solution(self, solution):
        """
        Evaluates a solution on the problem and returns an array indicating which clauses are satisfied

        Arguments
        solution (list of int): The solution in the form [-1, 3, 5, -6, ...]

        Returns
        (np.ndarray of bool): An array of length m which clauses are satisfied

        """
        return np.array([any([assignment in clause for assignment in solution]) for clause in self.clauses])

    def score_solution(self, solution):
        return sum(self.evaluate_solution(solution))

    def clauses_with_variables(self, variables):
        return_clauses = []
        for variable in variables:
            for clause in self.clauses:
                if variable in clause:
                    return_clauses.append(clause)
                if -1 * variable in clause:
                    return_clauses.append(clause)
        return return_clauses

    def __repr__(self):
        return f'SAT(n = {self.n}, m = {self.m}, k = {self.k}, name = \'{self.name}\')'


def read_sat_problem(filename):
    """
    Reads in a .cnf or .cnf.lzma file by uncompressing it and parsing the clauses.
    
    Arguments
    filename (str): The path to the file to read
    
    Returns
    n (int): The number of variables in the problem
    m (int): The number of clauses in the problem
    clauses (np.ndarray): The conjunctive normal form clauses
    """

    clauses = []

    if filename[-5:] == '.lzma':
        file = lzma.open(filename, mode='rb')
    elif filename[-4:] == '.cnf':
        file = open(filename, mode='rb')
    else:
        raise ValueError(f'Unknown filetype for file {filename}')

    for line in file:

        line = line.decode("utf-8")
        if line[0] == 'c' or line[0] == 'p':
            continue
        else:
            words = line.split(" ")

            clauses.append([int(variable) for variable in words if re.match('-?[1-9]+[0-9]*', variable)])

    file.close()

    return SATProblem(np.array(clauses, dtype=object), name=filename.split('/')[-1])


def bitstring_to_intarray(bitstring: str):
    intarray = []
    for i in range(len(bitstring)):
        if bitstring[i] == '0':
            intarray.append(-1 * (i+1))
        else:
            intarray.append(i+1)

    return intarray
