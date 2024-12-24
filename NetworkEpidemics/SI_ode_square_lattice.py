import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# SI Model on a Square Lattice (ODE Version)
# Author: Antonio Scala
# Course: Complex Systems (Master Level)


# ------------------------------
# PARAMETERS
# ------------------------------

L = 50             # Lattice size (L x L)
beta = 0.5         # Infection rate
t_max = 20         # Total simulation time
timesteps = 200    # Number of time points
initial_infected_fraction = 0.01  # Fraction of initially infected nodes

# ------------------------------
# FUNCTIONS
# ------------------------------

def initialize_lattice(L, initial_infected_fraction):
    """
    Initialize the lattice with fractions of susceptible (s) and infected (i) individuals.
    Returns:
        s: Susceptible fraction (LxL array)
        i: Infected fraction (LxL array)
    """
    i = np.zeros((L, L))
    num_initial_infected = int(L * L * initial_infected_fraction)
    
    # Randomly infect some nodes
    for _ in range(num_initial_infected):
        x, y = np.random.randint(0, L, size=2)
        i[x, y] = 1
    
    s = 1 - i  # Susceptible fraction
    return s, i

def si_model_with_neighbors(t, y, L, beta):
    """
    SI model on a square lattice with neighbor interactions.
    Arguments:
        t: Time (required by solver, but unused explicitly)
        y: Flattened array of susceptible (s) and infected (i) states
        L: Lattice size
        beta: Infection rate
    Returns:
        Flattened derivatives of s and i for all nodes
    """
    s = y[:L*L].reshape((L, L))  # Reshape s from flattened form
    i = y[L*L:].reshape((L, L))  # Reshape i from flattened form

    # Infection from neighbors
    neighbor_infection = (
        np.roll(i, 1, axis=0) +  # Top neighbor
        np.roll(i, -1, axis=0) + # Bottom neighbor
        np.roll(i, 1, axis=1) +  # Left neighbor
        np.roll(i, -1, axis=1)   # Right neighbor
    )
    
    # Compute infection dynamics
    ds_dt = -beta * s * neighbor_infection
    di_dt = beta * s * neighbor_infection

    # Return flattened derivatives
    return np.concatenate([ds_dt.flatten(), di_dt.flatten()])

def simulate_si_with_neighbors(L, beta, t_max, timesteps, initial_infected_fraction):
    """
    Simulate the SI model on a square lattice with neighbor interactions.
    """
    # Initialize lattice
    s, i = initialize_lattice(L, initial_infected_fraction)
    
    # Flatten initial conditions for solver
    y0 = np.concatenate([s.flatten(), i.flatten()])
    
    # Time points for integration
    t_eval = np.linspace(0, t_max, timesteps)
    
    # Solve the PDEs using scipy.integrate.solve_ivp
    solution = solve_ivp(
        si_model_with_neighbors,        # The SI model with neighbor interactions
        [0, t_max],                     # Time span
        y0,                             # Initial conditions
        t_eval=t_eval,                  # Time points to evaluate
        args=(L, beta),                 # Additional arguments to pass
        method='RK45'                   # Runge-Kutta method for numerical integration
    )
    
    # Reshape results into (timesteps, L, L) arrays for s and i
    s_values = solution.y[:L*L, :].T.reshape((timesteps, L, L))
    i_values = solution.y[L*L:, :].T.reshape((timesteps, L, L))
    t_values = solution.t
    return t_values, s_values, i_values

# ------------------------------
# VISUALIZATION
# ------------------------------

def animate_si_lattice(t_values, i_values):
    """
    Animate the infected lattice over time.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(i_values[0], cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title("Infected Fraction - Timestep 0")
    ax.axis('off')

    def update(frame):
        im.set_array(i_values[frame])
        ax.set_title(f"Infected Fraction - Timestep {frame}")
        return [im]

    ani = FuncAnimation(fig, update, frames=len(t_values), interval=100, repeat=False)
    plt.show()

# ------------------------------
# RUN SIMULATION
# ------------------------------

# Run the SI model on a lattice
t_values, s_values, i_values = simulate_si_with_neighbors(
    L=L,
    beta=beta,
    t_max=t_max,
    timesteps=timesteps,
    initial_infected_fraction=initial_infected_fraction
)

# Visualize the infected fraction
animate_si_lattice(t_values, i_values)
