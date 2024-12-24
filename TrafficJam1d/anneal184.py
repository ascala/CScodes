import numpy as np
import matplotlib.pyplot as plt
import wolfram_rules as wolf

# Parameters for the simulation
lane_length = 50  # Length of the lane
thermalization_steps = 200  # Number of steps to discard for thermalization
steps = 100  # Number of time steps to simulate

if False:
    n1=int(lane_length*0.9) # initial number of cars
else:
    n1=0  # initial number of cars

# Create a vector with the appropriate number of 1's and 0's
lane = np.array([1] * n1 + [0] * (lane_length - n1))
# Shuffle the vector to randomly distribute 1's and 0's
np.random.shuffle(lane)

# Initialize lists to store results
avg_speeds = []
speed_variances = []
densities=[]

# Run simulations for each arrival rate
#for k in range(1, n1, 1):
#    lane[np.random.choice(np.where(lane == 1)[0])] = 0 # take out a car
for k in range(1, n1, 1):
    lane[np.random.choice(np.where(lane == 0)[0])] = 1 # put in a car
    n_cars=sum(lane) # number of cars
    densities.append(n_cars/lane_length)  
    print(n_cars)  
    
    # thermalization steps to discard
    lane, speeds = wolf.run184(thermalization_steps, lane)
    #acquisition steps to collect data
    lane, speeds = wolf.run184(steps, lane)

    # Calculate averages and variances after discarding the first 50 timesteps
    avg_speed = np.mean(speeds)
    speed_variance = np.var(speeds)
    
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
