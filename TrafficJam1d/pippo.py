import numpy as np
import matplotlib.pyplot as plt

def update_plot(evolution):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    
    # Initialize an empty array for the plot
    current_data = []

    for step in evolution:
        current_data.append(step)  # Add the new step
        ax.clear()  # Clear the previous plot
        ax.imshow(current_data, cmap='binary', interpolation='nearest')
        ax.set_title("Cellular Automata Evolution")
        ax.set_xlabel("Cells")
        ax.set_ylabel("Time Steps")
        plt.pause(0.5)  # Pause to allow the plot to update

    plt.ioff()  # Turn off interactive mode
    plt.show()

# Example usage with a sample evolution data
evolution = [
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1]
]

update_plot(evolution)
