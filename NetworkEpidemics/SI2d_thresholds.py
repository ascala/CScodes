import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# PARAMETERS
# ------------------------------
beta = 1.0  # Infection rate
gamma = 0.75  # Recovery rate
C = np.array([[1.0, 0.2],  # Connectivity matrix
              [0.2, 1.0]])

# ------------------------------
# FUNCTIONS
# ------------------------------

def compute_matrix(s1, s2):
    """
    Compute the matrix M - gamma I given s1 and s2.
    Args:
        s1, s2: Values of s_k.
    Returns:
        M - gamma I matrix.
    """
    S = np.diag([s1, s2])  # Diagonal matrix of s_k
    M = beta * S @ C  # M = beta * diag(s) * C
    return M - gamma * np.eye(2)

def is_negative_definite(matrix):
    """
    Check if a matrix is negative definite.
    Args:
        matrix: A square matrix.
    Returns:
        True if the matrix is negative definite, False otherwise.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)  # Compute eigenvalues
    return np.all(eigenvalues < 0)

# ------------------------------
# HEATMAP GENERATION
# ------------------------------

# Create a grid of s1 and s2 values
s1_values = np.linspace(0.01, 0.99, 100)  # Avoid 0 and 1 for valid log/ops
s2_values = np.linspace(0.01, 0.99, 100)
s1_grid, s2_grid = np.meshgrid(s1_values, s2_values)

# Check negative definiteness for each point in the grid
negative_definite_map = np.zeros_like(s1_grid, dtype=bool)

for i in range(s1_grid.shape[0]):
    for j in range(s1_grid.shape[1]):
        s1, s2 = s1_grid[i, j], s2_grid[i, j]
        M_minus_gamma = compute_matrix(s1, s2)
        negative_definite_map[i, j] = is_negative_definite(M_minus_gamma)

# ------------------------------
# PLOTTING
# ------------------------------

plt.figure(figsize=(8, 6))
plt.imshow(1-negative_definite_map, extent=(0, 1, 0, 1), origin='lower', cmap='coolwarm', aspect='auto')
plt.colorbar(label="Negative Definite (1=True, 0=False)")
plt.xlabel("$s_1$")
plt.ylabel("$s_2$")
plt.title("Heatmap of Negative Definiteness for $M - \gamma I$")
plt.grid(False)
plt.show()
