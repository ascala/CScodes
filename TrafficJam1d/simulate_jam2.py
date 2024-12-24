import numpy as np
import matplotlib.pyplot as plt
import jam2 as jam

# Parameters for the simulation
L=100
lane_length = L  # Length of the lane
thermalization_steps = 4*L  # Number of steps to discard for thermalization
steps = L  # Number of time steps to simulate
lambda_ = 1/lane_length # jamming rate
arrival_rates = lambda_*np.arange(1e-1, 7e-1, 1e-3)  # Arrival rates from 0.01 to 0.1

# Initialize lists to store results
avg_densities = []
avg_speeds = []
density_variances = []
speed_variances = []

# Run simulations for each arrival rate
lane = np.zeros(lane_length, dtype=int)  # Initialize the lane
for arrival_rate in arrival_rates:
    #lane = np.zeros(lane_length, dtype=int)  # Initialize the lane
    
    # thermalization steps to discard
    lane, densities, speeds = jam.run_simulation(thermalization_steps, arrival_rate, lane)
    #acquisition steps to collect data
    lane, densities, speeds = jam.run_simulation(steps, arrival_rate, lane)

    # Calculate averages and variances after discarding the first 50 timesteps
    avg_density = np.mean(densities)
    avg_speed = np.mean(speeds)
    density_variance = np.var(densities)
    speed_variance = np.var(speeds)

    # Store the results
    avg_densities.append(avg_density)
    avg_speeds.append(avg_speed)
    density_variances.append(density_variance)
    speed_variances.append(speed_variance)

# Convert lists to numpy arrays for plotting
avg_densities = np.array(avg_densities)
avg_speeds = np.array(avg_speeds)
density_variances = np.array(density_variances)
speed_variances = np.array(speed_variances)

# Plot average density vs arrival rate with error bars
plt.figure(figsize=(8, 12))

plt.subplot(3, 2, 1)
plt.plot(lane_length*arrival_rates, avg_densities, color='b', marker='o', linestyle='', label='Simulation')
#plt.errorbar(lane_length*arrival_rates, avg_densities, yerr=np.sqrt(density_variances), color='b', fmt='o', capsize=5,label='Simulation')
#theoretical_densities = arrival_rates/(1+arrival_rates)
theoretical_densities = np.minimum(lane_length*arrival_rates,1)
#plt.plot(arrival_rates,avg_densities,marker='o',color='b')
plt.plot(lane_length*arrival_rates, theoretical_densities, linestyle='--', color='r',label='Theoretical')
plt.xlabel('Arrival Rate')
plt.ylabel('Average Density')
#plt.title('Average Density vs Arrival Rate with Error Bars')
plt.legend()  # Add a legend to differentiate between the plots
plt.grid()

plt.subplot(3, 2, 2)
plt.plot(lane_length*arrival_rates, density_variances, color='b', marker='o', linestyle='', label='Simulation')
plt.xlabel('Arrival Rate')
plt.ylabel('Density Variance')
#plt.title('Average Density vs Arrival Rate with Error Bars')
plt.legend()  # Add a legend to differentiate between the plots
plt.grid()


# Plot average speed vs arrival rate with error bars
plt.subplot(3, 2, 3)
plt.plot(lane_length*arrival_rates, avg_speeds, color='r', marker='o', linestyle='', label='Simulation')
#plt.errorbar(lane_length*arrival_rates, avg_speeds, yerr=np.sqrt(speed_variances), color='r', fmt='o', capsize=5,label='Simulation')
theoretical_speeds=(1-theoretical_densities)#**2
#plt.plot(arrival_rates,avg_speeds,marker='o',color='r')
plt.plot(lane_length*arrival_rates, theoretical_speeds, linestyle='--', color='b',label='Theoretical')
plt.xlabel('Arrival Rate')
plt.ylabel('Average Speed')
#plt.title('Average Speed vs Arrival Rate with Error Bars')
plt.legend()  # Add a legend to differentiate between the plots
plt.grid()


plt.subplot(3, 2, 4)
plt.plot(lane_length*arrival_rates, speed_variances, color='b', marker='o', linestyle='', label='Simulation')
plt.xlabel('Arrival Rate')
plt.ylabel('Speed Variance')
#plt.title('Average Density vs Arrival Rate with Error Bars')
plt.legend()  # Add a legend to differentiate between the plots
plt.grid()


# Plot average speed vs average density
plt.subplot(3, 1, 3)
theoretical_speeds=(1-avg_densities)**2
plt.plot(avg_densities, avg_speeds, marker='o', linestyle='', color='b',label='Observed')
plt.plot(avg_densities, theoretical_speeds, marker='+', linestyle='', color='r',label='Theoretical')
plt.xlabel('Average Density')
plt.ylabel('Average Speed')
#plt.title('Average Speed vs Arrival Rate with Error Bars')
plt.legend()  # Add a legend to differentiate between the plots
plt.grid()


plt.tight_layout()
plt.show()
