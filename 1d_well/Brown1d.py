import numpy as np # Numerical computation library

# Brownian simulation 
def Brown1d_simulation(x_init, n_steps, sigma, dt, V, dV_dx):
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


