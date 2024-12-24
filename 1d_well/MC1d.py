import numpy as np


# Monte Carlo simulation
def monte_carlo_1d_simulation(x_init, n_steps, beta, step_size, V):
    x = x_init  # Start at initial position
    positions = []  # Store accepted positions for analysis

    for step in range(n_steps):
        # Propose a new position x_new from a uniform distribution around x
        x_new = x + step_size * (2 * np.random.rand() - 1)

        # Compute the change in potential energy
        delta_V = V(x_new) - V(x)

        # Decide whether to accept the move using the Metropolis criterion
        if np.random.rand() < np.exp(-beta * delta_V):
            x = x_new  # Accept the move

        # Store the current position
        positions.append(x)

    return np.array(positions)
