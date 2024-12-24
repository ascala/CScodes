import numpy as np
import matplotlib.pyplot as plt

# Function to simulate one time step
def step(lane, arrival_rate):
    lane_length=len(lane)
    
    

    # Update the lane with new arrivals    
    if np.random.rand() <arrival_rate and lane[0] == 0:  # Only place a car if the position is empty
        lane[0] = 1

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
def run_simulation(steps, arrival_rate, lane):
    densities = []
    speeds = []

    for step_num in range(steps):
        lane, cars_moved = step(lane, arrival_rate)
        n_cars=np.sum(lane)
        density = n_cars / len(lane)  # Calculate density as the fraction of occupied cells
        speed = cars_moved / n_cars   # Calculate speed as the fraction of cars that moved
        densities.append(density)
        speeds.append(speed)

    return lane, densities, speeds
