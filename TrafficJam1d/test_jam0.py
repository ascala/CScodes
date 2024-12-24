import numpy as np
import matplotlib.pyplot as plt

import jam0 as jam

# Parameters for the simulation
lane_length = 50  # Length of the lane
arrival_rate = 1.1  # Arrival rate of cars (0 to 1)
critical_rate = 1.0  # Critical arrival rate for jamming
steps = 100  # Number of time steps to simulate


# Initialize the lane
lane = np.zeros(lane_length, dtype=int)  # Create a lane filled with zeros (empty)

# Run the simulation and collect density and speed data over time
lane, densities, speeds = jam.run_simulation(steps, arrival_rate, lane)

# Calculate theoretical stationary density and velocity
theoretical_density = lane_length*arrival_rate/(1+lane_length*arrival_rate)
theoretical_speed = (1-theoretical_density)
print(theoretical_density)
print(theoretical_speed)

# Plot density vs time
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(range(steps), np.array(densities), marker='o')
plt.xlabel('Time Step')
plt.ylabel('Density')
plt.title('Density vs Time in Single Lane Traffic Simulation')
plt.axhline(y=theoretical_density, color='r', linestyle='--', label='Theoretical')
plt.grid()

# Plot speed vs time
plt.subplot(2, 1, 2)
plt.plot(range(steps), speeds, marker='o', color='r')
plt.axhline(y=theoretical_speed, color='b', linestyle='--', label='Theoretical')
plt.xlabel('Time Step')
plt.ylabel('Speed (Fraction of Cars Moving)')
plt.title('Speed vs Time in Single Lane Traffic Simulation')
plt.grid()

plt.tight_layout()
plt.show()
