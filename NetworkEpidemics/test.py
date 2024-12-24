import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# PARAMETERS
# ------------------------------
beta = 1.0  # Infection rate
gamma = 0.5  # Recovery rate
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

def classify_matrix(matrix):
    """
    Classify the matrix based on its eigenvalues:
        0: All eigenvalues negative (negative definite).
        1: Mixed eigenvalues (one positive, one negative).
        2: All eigenvalues non-negative or zero.
    Args:
        matrix: A square matrix.
    Returns:
        Classification code (0, 1, or 2).
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.all(eigenvalues < 0):
        return 0  # Negative definite
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return 1  # Mixed eigenvalues
    else:
        return 2  # Not negative definite

# ------------------------------
# HEATMAP GENERATION
# ------------------------------

# Create a grid of s1 and s2 values
s1_values = np.linspace(0.01, 0.99, 100)  # Avoid 0 and 1 for valid log/ops
s2_values = np.linspace(0.01, 0.99, 100)
s1_grid, s2_grid = np.meshgrid(s1_values, s2_values)

# Classification map: 0 (negative definite), 1 (mixed eigenvalues), 2 (not negative definite)
classification_map = np.zeros_like(s1_grid, dtype=int)

for i in range(s1_grid.shape[0]):
    for j in range(s1_grid.shape[1]):
        s1, s2 = s1_grid[i, j], s2_grid[i, j]
        M_minus_gamma = compute_matrix(s1, s2)
        classification_map[i, j] = classify_matrix(M_minus_gamma)

# ------------------------------
# PLOTTING
# ------------------------------

# Define a custom colormap for classification
from matplotlib.colors import ListedColormap

cmap = ListedColormap(["red", "yellow", "blue"])  # Red = 0, Yellow = 1, Blue = 2

plt.figure(figsize=(8, 6))
plt.imshow(classification_map, extent=(0, 1, 0, 1), origin='lower', cmap=cmap, aspect='auto')
cbar = plt.colorbar(ticks=[0, 1, 2])
cbar.ax.set_yticklabels(["Negative Definite", "Mixed Eigenvalues", "Not Negative Definite"])
plt.xlabel("$s_1$")
plt.ylabel("$s_2$")
plt.title("Matrix Classification Heatmap")
plt.grid(False)
plt.show()
