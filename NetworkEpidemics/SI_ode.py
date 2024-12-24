import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# SI Model using Differential Equations
# Author: Antonio Scala
# Course: Complex Systems (Master Level)

# ------------------------------
# PARAMETERS
# ------------------------------

beta = 0.3        # Infection rate
i0 = 0.01         # Initial fraction of infected individuals
s0 = 1 - i0       # Initial fraction of susceptible individuals
t_max = 50        # Total time to simulate
num_points = 1000 # Number of time points for integration

# ------------------------------
# SI MODEL EQUATIONS
# ------------------------------

def si_model(t, y, beta):
    """
    Differential equations for the SI model.
    Arguments:
        t: Time (required by solver, but unused explicitly)
        y: List containing [s, i] (fractions of susceptible and infected)
        beta: Infection rate
    Returns:
        List of derivatives [ds/dt, di/dt]
    """
    s, i = y
    ds_dt = -beta * s * i  # Change in susceptible population
    di_dt = beta * s * i   # Change in infected population
    return [ds_dt, di_dt]

# ------------------------------
# SOLVING THE DIFFERENTIAL EQUATIONS
# ------------------------------

# Initial conditions: [s0, i0]
y0 = [s0, i0]

# Time points where the solution is computed
t_eval = np.linspace(0, t_max, num_points)

# Solve the differential equations using scipy.integrate.solve_ivp
solution = solve_ivp(
    si_model,                      # The SI model equations
    [0, t_max],                    # Time span
    y0,                            # Initial conditions
    t_eval=t_eval,                 # Time points to evaluate the solution
    args=(beta,),                  # Additional arguments to pass (beta)
    method='RK45'                  # Runge-Kutta method for numerical integration
)

# Extract the solution: s(t) and i(t)
s_values = solution.y[0]  # Susceptible fraction
i_values = solution.y[1]  # Infected fraction
t_values = solution.t     # Time values

# ------------------------------
# PLOT THE RESULTS
# ------------------------------

def plot_si_model(t, s, i):
    """
    Plot the results of the SI model simulation.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(t, s, label='Susceptible (s)', color='blue')
    plt.plot(t, i, label='Infected (i)', color='red')
    plt.title('SI Model Dynamics')
    plt.xlabel('Time')
    plt.ylabel('Fraction of Population')
    plt.legend()
    plt.grid()
    plt.show()

# Plot the simulation results
plot_si_model(t_values, s_values, i_values)
