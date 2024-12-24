import numpy as np

import matplotlib.pyplot as plt

def compute_eigenvalues_and_eigenvectors(L):
    """
    Computes the eigenvalues and eigenvectors of the Laplacian matrix.

    Args:
        L (numpy.ndarray): Laplacian matrix (n x n).

    Returns:
        tuple: Eigenvalues (1D array) and eigenvectors (2D array).
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L) # eigh is optimised for symmetric matrices
    
    return eigenvalues, eigenvectors


def solve_laplacian(L, s):
    """
    Solves the equation L v = s using the pseudo-inverse of L.

    Args:
        L (numpy.ndarray): Laplacian matrix (n x n).
        s (numpy.ndarray): Source vector (n x 1).

    Returns:
        numpy.ndarray: Solution vector v (n x 1).
    """
    # Compute the pseudo-inverse of L
    L_pseudo_inverse = np.linalg.pinv(L)

    # Solve for v
    v = L_pseudo_inverse @ s

    return v

# Example Laplacian matrix and source vector
L = np.array([[ 3, -1, -2],
              [-1,  3, -2],
              [-2, -2,  4]])
print("Laplacian:")
print(L)

s = np.array([1, 0, -1])
print("\nSources:")
print(s)

# Solve L v = s
v = solve_laplacian(L, s)
print("\n Solution vector v:")
print(v)

print("\nCheck:")
print("L v - s = ",L@v-s)


# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = compute_eigenvalues_and_eigenvectors(L)

print("Eigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# Plot eigenvalues
plt.plot(sorted(eigenvalues), 'o-', label="Eigenvalues")
plt.title("Spectrum of the Laplacian")
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.grid()
plt.legend()
plt.show()
