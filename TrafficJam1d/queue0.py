import numpy as np

# Function to simulate one time step
def step(L, car_pos, waiting, arrival_rate, delta):
    
    if np.random.rand() < arrival_rate:  # Check if a new car arrives based on arrival rate
        waiting += 1  # Increase the waiting count
        
    if waiting > 0:  # Check if there are waiting cars
        if len(car_pos) == 0:  # If there are no cars in the lane
            car_pos = np.concatenate(([L], car_pos), dtype=int)  # Add the new car position L
            waiting -= 1  # Decrease the waiting count
        else:  # If there are already cars in the lane
            x_max = car_pos[0]  # Get the position of the last car
            if L > x_max+delta:  # Check if the new position L is greater than the first car's position
                car_pos = np.concatenate(([L], car_pos), dtype=int)  # Add the new car position L
                waiting -= 1  # Decrease the waiting count

    n = len(car_pos)
    new_pos = np.copy(car_pos)  # Create a copy of car_pos
    cars_moved = 0

    for i in range(0, n - 1):
        if car_pos[i] > car_pos[i + 1] + delta:  # Check if the current car can move
            new_pos[i] = car_pos[i] - 1  # Move the car by 1 unit
            cars_moved += 1

    if n > 0:
        new_pos[-1] -= 1  # Move the last car back by 1 unit
        cars_moved += 1
        if new_pos[-1] == 0:  # If the last car reaches position 0
            new_pos = new_pos[:-1]  # Remove it from the list

    return new_pos, waiting  # Return updated positions and waiting count
