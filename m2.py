from scipy.linalg import expm
import numpy as np

# https://learn.microsoft.com/en-us/azure/quantum/user-guide/libraries/chemistry/concepts/algorithms

# error is function of m, t, and n
# m is the number of hamiltonian's in the sum of one individual hamiltonian
# t is time
# n is the number of steps in the approximation
# for general trotter-suzuki, the function looks like 

# O(((m*t)^2k+1)/n^2k) 

# Things to do:

# 1. going to need a randomized matrix generator that creates M matricies for the generalized case.
# 2. need to make plots of computationaly resource expenditure at increased accuracy
# 3. Full write-up on the method/code
# 4. anything else?

# this file will simply be for m = 2 and predefined KNOWN commutative matricies. can be expanded to calculate for m sub hamiltonian's but wanted 
# the easiest version to work first

def suzuki_trotter_step(A, B, t, order, n):
    # trivial at this point to define order of 2, this is just from all the tutorials we've seen
    if order == 2:
        exp_A = expm(-1j * A * t / (2 * n))
        exp_B = expm(-1j * B * t / (2 * n))
        return exp_A @ exp_B @ exp_B @ exp_A
    # this recursive programming seems to work, first time using it like this. 
    else:
        s_k = (4 - 4 ** (1 / (2 * order - 1))) ** -1
        outer = suzuki_trotter_step(A, B, s_k * t, order - 2, n)
        inner = suzuki_trotter_step(A, B, (1 - 4 * s_k) * t, order - 2, n)
        return np.linalg.matrix_power(outer, 2) @ inner @ np.linalg.matrix_power(outer, 2)

def suzuki_trotter(A, B, t, order, n):
    if order % 2 != 0:
        raise ValueError("Suzuki-Trotter formula is only defined for even orders.")

    approx_exp = np.identity(len(A))
    for _ in range(n):
        approx_exp = approx_exp @ suzuki_trotter_step(A, B, t, order, n)

    return approx_exp

def commutator(A, B):
    a = np.dot(A, B) - np.dot(B, A)
    if a.all() == 0:
        print("A and B commute!")
    else:
        print("A and B do not commute!")

# Define matrices A and B
A = np.array([[0, 1], [1, 0]])  # Pauli X matrix
B = np.array([[0, -1j], [1j, 0]])  # Pauli Y matrix
commutator(A,B)
t = 5.0  # Total time for the evolution
order = 4  # Order of approximation (must be even)
n = 100  # Number of repetitions

# Compute the Suzuki-Trotter approximation
approx_exp = suzuki_trotter(A, B, t, order, n)
# have to add this part to match the operations of the unitary operator in the defined function
A = (-1j * A * t)
B = (-1j * B * t)
actual = expm(A + B)
print("Calculated Matrix")
print(actual)
print("Approximated Matrix")
print(approx_exp)
print(f"Total time: {t} seconds", f"\nOrder of Approximations: {order}", f"\nNumber of Repetitions: {n}") 
