import numpy as np
import matplotlib.pyplot as plt
#Import Monte Carlo routines and potential from MC1d.py
from MC1d import V,monte_carlo_1d_simulation   
# Import autocorrelation functions defined in autocorrelation.py
from autocorrelation import autocorrelation, autocorrelation_fft



# Parameters for the simulation
n_steps = 100000  # Total number of Monte Carlo steps
beta = 1.0        # Inverse temperature (1/kT)
x_init = 0.0      # Initial position of the particle
step_size = 1.0   # Proposal step size for the random walk

# Run the simulation
positions = monte_carlo_1d_simulation(x_init, n_steps, beta, step_size)


# first figure
fig1 = plt.figure()

# Plot the histogram of positions
plt.hist(positions, bins=100, density=True, alpha=0.6, label="Monte Carlo distribution")

# Plot the analytical Boltzmann distribution for comparison
x_vals = np.linspace(-5, 5, 100)
boltzmann_dist = np.exp(-beta * V(x_vals))
boltzmann_dist /= np.trapz(boltzmann_dist, x_vals)  # Normalize

plt.plot(x_vals, boltzmann_dist, 'r-', label="Analytical distribution")
plt.xlabel("Position (x)")
plt.ylabel("Probability density")
plt.legend()
plt.title("Monte Carlo simulation of a particle in 1D harmonic potential")
plt.show(block=False)

# second figure
fig2 = plt.figure()

# Calculate AutoCorrelation Function of particle positions
lags, acf = autocorrelation(positions, max_lag=70)

# Plot the ACF
plt.stem(lags, acf, use_line_collection=True)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation Function (ACF)")
plt.show(block=False)


# third figure
fig3 = plt.figure()

# Calculate AutoCorrelation Function of particle positions via FFT
#lags, acf = autocorrelation(positions, max_lag=50)
lags, acf_fft = autocorrelation_fft(positions, max_lag=70) # faster !!!

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
