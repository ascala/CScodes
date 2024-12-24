import numpy as np
import matplotlib.pyplot as plt

# Parameters for the simulation
lane_length = 50  # Length of the lane
arrival_rate = 0.3  # Arrival rate of cars (0 to 1)
critical_rate = 1.0  # Critical arrival rate for jamming
steps = 100  # Number of time steps to simulate

# define the theoretical stationary density
def rho_eq(arrival_rate):
    
    if arrival_rate < 1:
        return arrival_rate 
    else:
        return critical_rate


# Initialize the lane
lane = np.zeros(lane_length, dtype=int)  # Create a lane filled with zeros (empty)

# Function to simulate one time step
def step(lane, arrival_rate):
    # Determine how many cars arrive
    num_arrivals = np.random.binomial(lane_length, arrival_rate)  # Cars trying to enter
    arrivals = np.random.choice(lane_length, num_arrivals, replace=False)  # Random positions for arrivals

    # Update the lane with new arrivals
    for pos in arrivals:
        if lane[pos] == 0:  # Only place a car if the position is empty
            lane[pos] = 1

    # Move cars forward
    new_lane = np.copy(lane)  # Create a copy for updates
    cars_moved = 0  # Counter for cars that moved
    for i in range(lane_length - 1):
        if lane[i] == 1 and lane[i + 1] == 0:  # If there's a car and the next position is empty
            new_lane[i] = 0  # Leave the current position empty
            new_lane[i + 1] = 1  # Move the car forward
            cars_moved += 1

    # Handle the last position (cars exit the lane)
    if lane[lane_length - 1] == 1:  # If there's a car at the last position
        new_lane[lane_length - 1] = 0  # The car exits the lane
        cars_moved += 1

    return new_lane, cars_moved


# Function to run the simulation and collect density and speed data
def run_simulation(steps, arrival_rate):
    global lane
    densities = []
    speeds = []

    for step_num in range(steps):
        lane, cars_moved = step(lane, arrival_rate)
        density = np.sum(lane) / lane_length  # Calculate density as the fraction of occupied cells
        speed = cars_moved / lane_length  # Calculate speed as the fraction of cars that moved
        densities.append(density)
        speeds.append(speed)

    return densities, speeds

# Run the simulation and collect density and speed data over time
densities, speeds = run_simulation(steps, arrival_rate)

# Calculate the theoretical stationary density
theoretical_density = rho_eq(arrival_rate)

# Plot density vs time
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(range(steps), np.array(densities)-theoretical_density, marker='o')
plt.xlabel('Time Step')
plt.ylabel('Density')
plt.title('Density vs Time in Single Lane Traffic Simulation')
plt.grid()

# Plot speed vs time
plt.subplot(2, 1, 2)
plt.plot(range(steps), speeds, marker='o', color='r')
plt.xlabel('Time Step')
plt.ylabel('Speed (Fraction of Cars Moving)')
plt.title('Speed vs Time in Single Lane Traffic Simulation')
plt.grid()

plt.tight_layout()
plt.show()
