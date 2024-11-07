import numpy as np
import matplotlib.pyplot as plt

def gillespie_simulation(reactants, products, rates, initial_state, reactant_names, max_time):
    """
    Simulate a stochastic process using the Gillespie algorithm.

    Parameters:
    reactants (ndarray): 2D array of reactants for each reaction.
    products (ndarray): 2D array of products for each reaction.
    rates (list): List of reaction rates.
    initial_state (list): Initial state of the system.
    reactant_names (list): List of names for the reactants.
    max_time (float): Maximum simulation time.

    Returns:
    times (list): List of times at which events occurred.
    states (list): List of states at each time point.
    """
    state = np.array(initial_state)
    time = 0
    times = [time]
    states = [state.copy()]

    while time < max_time:
        # Calculate propensities based on current state
        propensities = rates * np.prod(state[None, :] ** reactants, axis=1)
        total_propensity = np.sum(propensities)

        if total_propensity == 0:
            break

        # Time to next reaction
        tau = np.random.exponential(1 / total_propensity)
        time += tau

        # Determine which reaction occurs
        reaction_index = np.random.choice(len(rates), p=propensities / total_propensity)
        
        # Update state based on the reaction
        state -= reactants[reaction_index]  # Decrease reactants
        state += products[reaction_index]   # Increase products

        times.append(time)
        states.append(state.copy())

    return times, states

# Example usage
reactants = np.array([[1, 1, 0],  # A + B
                      [0, 0, 2],  # 2C
                      [1, 0, 0],  # A
                      [0, 1, 0]]) # B

products = np.array([[0, 0, 2],  # 2C
                     [1, 1, 0],  # A + B
                     [0, 1, 0],  # B
                     [1, 0, 0]]) # A
# these structures are for the reactions: 
# A + B -> 2C
# 2C -> A + B
# A -> B
# B -> A

rates = [0.001, 0.0001, 0.01, 0.01]  # Reaction rates

initial_state = [20, 80, 40]  # Initial number of A, B, and C molecules
max_time = 100  # Maximum simulation time
reactant_names = ['A', 'B', 'C']  # Names for the reactants

times, states = gillespie_simulation(reactants, products, rates, initial_state, reactant_names, max_time)

# Plot results
states = np.array(states)
for i in range(len(reactant_names)):
    plt.plot(times, states[:, i], label=reactant_names[i])
plt.xlabel('Time')
plt.ylabel('Number of molecules')
plt.legend()
plt.show()
