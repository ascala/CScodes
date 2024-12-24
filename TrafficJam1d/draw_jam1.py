import numpy as np
import matplotlib.pyplot as plt

import jam1 as jam

# Parameters for the simulation
lane_length = 50  # Length of the lane
arrival_rate = 2/lane_length  # Arrival rate of cars (0 to 1)
critical_rate = 1.0  # Critical arrival rate for jamming
steps = 800  # Number of time steps to simulate before the visualization


# Initialize the lane
lane = np.zeros(lane_length, dtype=int)  # Create a lane filled with zeros (empty)

# Evolution of the system
evolution = []

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

# simulate steps before visualization
for step_num in range(steps):
    lane, cars_moved = jam.step(lane, arrival_rate)

# simulate and visualise
for step_num in range(lane_length):
    lane, cars_moved = jam.step(lane, arrival_rate)
    evolution.append(lane.copy())
    # Plot the evolution of the system
    ax.clear()  # Clear the previous plot
    ax.imshow(evolution, cmap='binary', interpolation='nearest')
    ax.set_title("Cellular Automata Evolution")
    ax.set_xlabel("Cells")
    ax.set_ylabel("Time Steps")
    plt.pause(0.01)  # Pause to allow the plot to update

plt.ioff()  # Turn off interactive mode
plt.show()
