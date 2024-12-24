import numpy as np
import matplotlib.pyplot as plt
import wolfram_rules as wolf

# Parameters for the simulation
lane_length = 100  # Length of the lane
thermalization_steps = 600  # Number of steps to discard for thermalization
steps = 300  # Number of time steps to simulate
densities = np.arange(2/lane_length, 9e-1, 1e-3)  

# Initialize lists to store results
avg_speeds = []
speed_variances = []

# Run simulations for each arrival rate
for density in densities:
    lane = np.random.choice([0, 1], size=lane_length, p=[1-density, density])  # Initialize the lane
    n_cars=sum(lane) # number of cars
    
    # thermalization steps to discard
    lane, speeds = wolf.run58336(thermalization_steps, lane)
    #acquisition steps to collect data
    lane, speeds = wolf.run58336(steps, lane)

    # Calculate averages and variances after discarding the first 50 timesteps
    avg_speed = np.mean(speeds)
    speed_variance = np.var(speeds)
    if density==densities[0]:
        print(speeds[0:3])
    
    # Store the results
    avg_speeds.append(avg_speed)
    speed_variances.append(speed_variance)

# Convert lists to numpy arrays for plotting
avg_speeds = np.array(avg_speeds)
speed_variances = np.array(speed_variances)

# Plot average density vs arrival rate with error bars
plt.figure(figsize=(12, 6))


# Plot average speed vs arrival rate with error bars
plt.subplot(2, 1, 2)
plt.errorbar(densities, avg_speeds, yerr=np.sqrt(speed_variances), color='r', fmt='o', capsize=5,label='Simulation')
plt.xlabel('Density')
plt.ylabel('Speed')
#plt.title('Speed vs Density with Error Bars')
plt.legend()  # Add a legend to differentiate between the plots
plt.grid()

plt.tight_layout()
plt.show()
