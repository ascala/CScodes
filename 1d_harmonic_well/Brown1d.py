import numpy as np

# Define the potential function V(x) and its derivative
def V(x):
    """Potential energy function."""
    return 0.5 * x**2  # Harmonic potential (V(x) = 0.5 * x^2)

def dV_dx(x):
    """Derivative of the potential energy function."""
    return x  # dV/dx = x for harmonic potential

# Brownian simulation 
def Brown1d_simulation(x_init, n_steps, sigma, dt):
    x = x_init  # Start at initial position
    positions = []  # Store positions for analysis

    for step in range(n_steps):
        # Compute drift and diffusion terms
        drift = -dV_dx(x) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal()  # Wiener process term
        
        # Update position
        x += drift + diffusion
        positions.append(x)

    return np.array(positions)
