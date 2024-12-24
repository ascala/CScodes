import numpy as np # Numerical computation library

import matplotlib.pyplot as plt # Plotting library

# Import the function to define the potential and its derivative
from Potential1d import Define_Potential

# Import the function for Brownian simulation defined in Brownian1d.py
from Brown1d import Brown1d_simulation

# Import autocorrelation functions defined in autocorrelation.py
from autocorrelation import autocorrelation, autocorrelation_fft


# Parameters for the simulation
n_steps = 1000  # Total number of time steps
beta = 1.0        # Inverse temperature (1/kT)
x_init = 0.0      # Initial position of the particle
dt = 0.01         # Time step size
sigma = np.sqrt(2 / beta)  # Diffusion coefficient

# Define the potential V(x) and the derivative dV/dx
V,dV_dx = Define_Potential()

# Run the simulation
positions = Brown1d_simulation(x_init, n_steps, sigma, dt, V, dV_dx)

#plt.style.use('science') # Use the scientific papers' custom style for plplot

fig1=plt.figure() # first figure

# Plot the histogram of positions
plt.hist(positions, bins=100, density=True, alpha=0.6, label="SDE distribution")

# Plot the analytical Boltzmann distribution for comparison
xmin=np.min(positions)
xmax=np.max(positions)
xrange=max(abs(xmin),abs(xmax)) 
x_vals = np.linspace(-xrange, xrange, 100)
boltzmann_dist = np.exp(-beta * V(x_vals))
boltzmann_dist /= np.trapz(boltzmann_dist, x_vals)  # Normalize

plt.plot(x_vals, boltzmann_dist, 'r-', label="Analytical distribution")
plt.xlabel("Position (x)")
plt.ylabel("Probability density")
plt.legend()
plt.title("SDE simulation of a particle in 1D harmonic potential")
plt.show(block=False)


# Calculate AutoCorrelation Function of particle positions via FFT
lags, acf_fft = autocorrelation_fft(positions, max_lag=100)  # faster !!!

fig2=plt.figure() # second figure

# Plot the FFT-calculated ACF
plt.stem(lags, acf_fft, use_line_collection=True)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation Function (ACF via FFT)")
plt.show(block=False)

# Allow time for the figures to render
plt.pause(0.1)
# Keep the script running so figures remain open
plt.ioff()  # Disable interactive mode
plt.show()

