from scipy.linalg import expm, norm
import numpy as np
import time
import matplotlib.pyplot as plt

# https://learn.microsoft.com/en-us/azure/quantum/user-guide/libraries/chemistry/concepts/algorithms

# error is function of m, t, and n
# m is the number of hamiltonian's in the sum of one individual hamiltonian
# t is time
# n is the number of steps in the approximation
# for general trotter-suzuki, the function looks like 

# O(((m*t)^2k+1)/n^2k) 

# Things to do:

# 1. Full write-up on the method/code
# 2. anything else?

def generate_random_hermitian_matrices(m, size, commutative=False):
    hermitian_matrices = []
    for _ in range(m):
        if commutative:
            # Diagonal matrices will always commute
            diagonal_elements = np.random.rand(size) + 1j * np.random.rand(size)
            hermitian_matrix = np.diag(diagonal_elements)
        else:
            # General case for random Hermitian matrices
            random_matrix = np.random.rand(size, size) + 1j * np.random.rand(size, size)
            hermitian_matrix = (random_matrix + random_matrix.conj().T) / 2
        
        hermitian_matrices.append(hermitian_matrix)
    
    return hermitian_matrices

def suzuki_trotter_step(matrices, t, order, n):
    if order == 2:
        result = np.identity(len(matrices[0]))
        for matrix in matrices:
            result = result @ expm(-1j * matrix * t / (2 * n))
        for matrix in reversed(matrices):
            result = result @ expm(-1j * matrix * t / (2 * n))
        return result
    else:
        s_k = (4 - 4 ** (1 / (2 * order - 1))) ** -1
        outer = suzuki_trotter_step(matrices, s_k * t, order - 2, n)
        inner = suzuki_trotter_step(matrices, (1 - 4 * s_k) * t, order - 2, n)
        return np.linalg.matrix_power(outer, 2) @ inner @ np.linalg.matrix_power(outer, 2)

def suzuki_trotter(matrices, t, order, n):
    if order % 2 != 0:
        raise ValueError("Suzuki-Trotter formula is only defined for even orders.")
    
    approx_exp = np.identity(len(matrices[0]))
    for _ in range(n):
        approx_exp = approx_exp @ suzuki_trotter_step(matrices, t, order, n)
    
    return approx_exp

def check_commutativity(matrices):
    n = len(matrices)
    for i in range(n):
        for j in range(i+1, n):  # Start from i+1 to avoid repeating pairs
            comm = np.dot(matrices[i], matrices[j]) - np.dot(matrices[j], matrices[i])
            if not np.allclose(comm, np.zeros_like(comm)):
                print(f"Matrices {i} and {j} do not commute.")
                return  # Exit after finding the first non-commuting pair
    print("All matrices commute!")

def actual_exp(matrices, t):
    total_matrix = np.zeros_like(matrices[0], dtype=np.complex_)
    for matrix in matrices:
        total_matrix += (-1j * matrix * t)
    return expm(total_matrix)

def main(m, size, time, order, n_steps):
    """
    Generate m matrices, check commutivity, calculate suzuki-trotter approximation, calculate the actual, compare the two

    :param m: Number of Hermitian matrices to generate.
    :param size: Size of the matrices (size x size).
    :param t: Number of seconds to evolve matrix
    :param order: order of the approximation
    :param n_steps: number of iterations 
    """
    matrices = generate_random_hermitian_matrices(m, size, False)
    check_commutativity(matrices)
    approx_exp = suzuki_trotter(matrices, time, order, n_steps)
    actual = actual_exp(matrices, time)
    print("Calculated Matrix")
    print(actual)
    print("Approximated Matrix")
    print(approx_exp)
    print(f"Total time: {time} seconds", f"\nOrder of Approximations: {order}", f"\nNumber of Repetitions: {n_steps}") 

def plots(matrices, order, t, steps):
    """
    Generate plots for a given list of matrices, a specified order of approximation, and a time evolution value of t

    :param matrices: List of sub-hamiltonian hermitian matrices.
    :param t: Number of seconds to evolve matrix
    :param order: order of the approximation
    :param steps: number of iterations to take
    """
    computation_times = []
    accuracies = []

    steps = range(steps)

    for i in steps:
        start_time = time.time()
        approx_exp = suzuki_trotter(matrices, t, order, i)
        end_time = time.time()
        actual = actual_exp(matrices, t)
        computation_time = end_time - start_time

        accuracy = norm(actual - approx_exp, 'fro')
        accuracy = float(100 - accuracy)
        computation_times.append(computation_time)
        accuracies.append(accuracy)

    plt.scatter(computation_times, accuracies)
    plt.xlabel('Computation Time (seconds)')
    plt.ylabel('Accuracy in percent')
    plt.title('Suzuki-Trotter Approximation: Accuracy vs Computation Time')
    #plt.yscale('log')
    plt.show()

m = 10
size = 2
order = 2
t = 10.0 
steps = 100
matrices = generate_random_hermitian_matrices(m, size, False)
plots(matrices, order, t, steps)
