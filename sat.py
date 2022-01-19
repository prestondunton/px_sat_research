import lzma
import numpy as np
import re


def get_unique_vars(clauses):
    unique_vars = set()
    for clause in clauses:
        for variable in clause:
            unique_vars.add(abs(variable))

    return unique_vars


class SATProblem:
    def __init__(self, clauses, name=None):
        self.name = name
        self.clauses = clauses
        self.unique_vars = get_unique_vars(clauses)
        self.var_to_clause_dict = None

        self.n = len(self.unique_vars)
        self.m = len(clauses)
        self.k = max([len(clause) for clause in clauses])

    def sat_by_variable(self, solution):
        """
        Returns a list for each clause specifying which variable assignments clauses the clause to be satisfied

        Arguments
        solution (iterable): The solution to validate clauses against.  Usually an array of integers.
        """
        solution_set = set(solution)
        return np.array([{variable for variable in clause if variable in solution_set} for clause in self.clauses])

    def evaluate_solution(self, solution):
        """
        Evaluates a solution on the problem and returns an array indicating which clauses are satisfied

        Arguments
        solution (list of int): The solution in the form [-1, 3, 5, -6, ...]

        Returns
        (np.ndarray of bool): An array of length m which clauses are satisfied

        """

        solution_set = set(solution)
        clause_satisfied = [False for _ in range(self.m)]
        for i in range(self.m):
            for variable in self.clauses[i]:
                if variable in solution_set:
                    clause_satisfied[i] = True
                    break
        return np.array(clause_satisfied)

    def score_solution(self, solution):

        return sum(self.evaluate_solution(solution))

    def clauses_with_variables(self, variables):
        """
        Takes an iterable of variables are returns the indices of self.clauses
        where those variables appear

        Arguments
        variables (iterable): The variables to filter with
        """
        if self.var_to_clause_dict is None:
            self.create_var_to_clause_dict()

        clauses = []
        for variable in variables:
            clauses += self.var_to_clause_dict[abs(variable)]

        return set(clauses)

    def create_var_to_clause_dict(self):
        """
        Creates a dictionary of all clauses where a variable appears
        self.var_to_clause_dict[3] has the indices into self.clauses where variable 3 appears
        """
        var_to_clause = {}
        for i in range(self.m):
            for variable in self.clauses[i]:
                if abs(variable) not in var_to_clause.keys():
                    var_to_clause[abs(variable)] = [i]
                else:
                    var_to_clause[abs(variable)].append(i)

        self.var_to_clause_dict = var_to_clause

    def __repr__(self):
        return f'SAT(n = {self.n}, m = {self.m}, k = {self.k}, name = \'{self.name}\')'


def read_sat_problem(filename):
    """
    Reads in a .cnf or .cnf.lzma file by uncompressing it and parsing the clauses.
    
    Arguments
    filename (str): The path to the file to read
    
    Returns
    sat (SATProblem): The SAT Problem object
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

            clauses.append({int(variable) for variable in words if re.match('-?[1-9]+[0-9]*', variable)})

    file.close()

    return SATProblem(np.array(clauses, dtype=object), name=filename.split('/')[-1])


def bitstring_to_int_array(bitstring: str):
    intarray = []
    for i in range(len(bitstring)):
        if bitstring[i] == '0':
            intarray.append(-1 * (i+1))
        else:
            intarray.append(i+1)

    return intarray


def get_assignments(solution, variables):
    variable_set = set([variable for variable in variables] + [-1 * variable for variable in variables])
    solution_set = set(solution)

    return solution_set.intersection(variable_set)
