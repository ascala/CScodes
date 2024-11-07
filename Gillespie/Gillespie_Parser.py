import re
import numpy as np
import matplotlib.pyplot as plt

def parse_reactions(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    rates = []
    reactants = []
    products = []
    reactant_names = set()

    for line in lines:
        # Extract the rate (including scientific notation)
        rate_match = re.match(r'([+-]?\d*\.?\d+([eE][+-]?\d+)?)', line)
        if rate_match:
            rate = float(rate_match.group(1))
            rates.append(rate)

        # Remove the rate part from the line
        line = line[rate_match.end():].strip()

        # Split the reaction into reactants and products
        reactants_part, products_part = line.split('->')

        # Normalize reactants and products by removing extra spaces around '+'
        reactants_part = re.sub(r'\s*\+\s*', '+', reactants_part)
        products_part = re.sub(r'\s*\+\s*', '+', products_part)

        # Extract reactants
        reactants_dict = {}
        for match in re.finditer(r'(\d*)\s*([A-Za-z]+)', reactants_part):
            count = int(match.group(1)) if match.group(1) else 1
            name = match.group(2)
            reactants_dict[name] = count
            reactant_names.add(name)

        # Extract products
        products_dict = {}
        for match in re.finditer(r'(\d*)\s*([A-Za-z]+)', products_part):
            count = int(match.group(1)) if match.group(1) else 1
            name = match.group(2)
            products_dict[name] = count
            reactant_names.add(name)

        reactants.append(reactants_dict)
        products.append(products_dict)

    # Create a sorted list of unique reactant names
    reactant_names = sorted(reactant_names)

    # Convert reactions to matrix form
    num_reactions = len(rates)
    num_reactants = len(reactant_names)

    reactants_matrix = np.zeros((num_reactions, num_reactants), dtype=int)
    products_matrix = np.zeros((num_reactions, num_reactants), dtype=int)

    for i, (react_dict, prod_dict) in enumerate(zip(reactants, products)):
        for name, count in react_dict.items():
            index = reactant_names.index(name)
            reactants_matrix[i, index] = count
        for name, count in prod_dict.items():
            index = reactant_names.index(name)
            products_matrix[i, index] = count

    return rates, reactants_matrix, products_matrix, reactant_names

def gillespie_simulation(reactants, products, rates, initial_state, max_time):
    """
    Simulate a stochastic process using the Gillespie algorithm.

    Parameters:
    reactants (ndarray): 2D array of reactants for each reaction.
    products (ndarray): 2D array of products for each reaction.
    rates (list): List of reaction rates.
    initial_state (list): Initial state of the system.
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


# Integrate parser and simulator
file_path = 'reactions.txt'  # Path to your reactions file
rates, reactants_matrix, products_matrix, reactant_names = parse_reactions(file_path)

# Output of the parser
print("Rates:", rates)
print("Reactants Matrix:\n", reactants_matrix)
print("Products Matrix:\n", products_matrix)
print("Reactant Names:", reactant_names)

# Initial state and maximum simulation time
initial_state = [20, 80, 40]  # Initial number of A, B, and C molecules
max_time = 100  # Maximum simulation time

# Run Gillespie simulation
times, states = gillespie_simulation(reactants_matrix, products_matrix, rates, initial_state, max_time)

# Plot results
states = np.array(states)
for i in range(len(reactant_names)):
    plt.plot(times, states[:, i], label=reactant_names[i])
plt.xlabel('Time')
plt.ylabel('Number of molecules')
plt.legend()
plt.show()
