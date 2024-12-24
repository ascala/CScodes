import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation

# SI Model on a Square Lattice
# Author: Antonio Scala
# Course: Complex Systems (Master Level)

# ------------------------------
# PARAMETERS
# ------------------------------

L = 50            # Lattice size (L x L)
beta = 0.3        # Infection probability (per neighbor, per time step)
initial_infected = 5  # Number of initially infected individuals
timesteps = 100   # Number of time steps to simulate

# ------------------------------
# FUNCTIONS
# ------------------------------

def initialize_lattice(L, initial_infected):
    """
    Initialize the lattice with all individuals susceptible (0)
    and a few randomly chosen individuals infected (1).
    """
    lattice = np.zeros((L, L), dtype=int)  # 0: Susceptible, 1: Infected
    for _ in range(initial_infected):
        x, y = np.random.randint(0, L, size=2)
        lattice[x, y] = 1
    return lattice

def infect_neighbors(lattice, x, y, beta):
    """
    Attempt to infect the neighbors of an infected cell (x, y) with probability beta.
    """
    L = lattice.shape[0]
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  # Neighboring cells (von Neumann neighborhood)
    
    for nx, ny in neighbors:
        if 0 <= nx < L and 0 <= ny < L:  # Check if within bounds
            if lattice[nx, ny] == 0:  # Susceptible
                if random.random() < beta:
                    lattice[nx, ny] = 1  # Infect

def update_lattice(lattice, beta):
    """
    Update the lattice for one time step: infect neighbors of all currently infected cells.
    """
    L = lattice.shape[0]
    infected_indices = np.argwhere(lattice == 1)  # Find all infected cells
    for x, y in infected_indices:
        infect_neighbors(lattice, x, y, beta)
    return lattice

def visualize_lattice(lattice, timestep):
    """
    Visualize the current state of the lattice.
    """
    plt.imshow(lattice, cmap='viridis', interpolation='nearest')
    plt.title(f"SI Model - Timestep {timestep}")
    plt.colorbar(label='State (0: Susceptible, 1: Infected)')
    plt.axis('off')
    plt.show()

# ------------------------------
# MAIN SIMULATION
# ------------------------------

def simulate_SI(L, beta, initial_infected, timesteps):
    """
    Simulate the SI model over a given number of time steps.
    """
    lattice = initialize_lattice(L, initial_infected)  # Initialize lattice
    states_over_time = [lattice.copy()]  # Store lattice states over time for visualization
    
    for t in range(timesteps):
        lattice = update_lattice(lattice, beta)  # Update lattice state
        states_over_time.append(lattice.copy())  # Record state
    
    return states_over_time

# ------------------------------
# RUN SIMULATION
# ------------------------------

# Run the SI model simulation
states_over_time = simulate_SI(L, beta, initial_infected, timesteps)

# ------------------------------
# ANIMATION
# ------------------------------

def animate_simulation(states_over_time):
    """
    Animate the simulation results over time.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(states_over_time[0], cmap='viridis', interpolation='nearest')
    ax.axis('off')

    def update(frame):
        im.set_array(states_over_time[frame])
        ax.set_title(f"Timestep {frame}")
        return [im]

    ani = FuncAnimation(fig, update, frames=len(states_over_time), interval=100, repeat=False)
    plt.show()

# Visualize animation
animate_simulation(states_over_time)
