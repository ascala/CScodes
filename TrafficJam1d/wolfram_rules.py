import numpy as np
import matplotlib.pyplot as plt

def step184(lane):
    new_lane = np.copy(lane)  # Create a copy for updates
    cars_moved = np.zeros(len(lane), dtype=int)  # Counter for cars that moved
    for i in range(len(lane) - 1):
        if lane[i] == 1 and lane[i + 1] == 0:
            new_lane[i] = 0
            new_lane[i + 1] = 1
            cars_moved[i] = 1
    if lane[-1] == 1 and lane[0] == 0:
        new_lane[-1] = 0
        new_lane[0] = 1
        cars_moved[-1] = 1
    return lane, cars_moved

def step58336(lane):
    new_lane = np.copy(lane)  # Create a copy for updates
    cars_moved = np.zeros(len(lane), dtype=int)  # Counter for cars that moved
    last=-1; nextlast=len(lane)-1;
    for i in range(len(lane) - 2):
        if lane[i] == 1 and lane[i + 1] + lane[i+2] == 0:
            new_lane[i] = 0
            new_lane[i + 1] = 1
            cars_moved[i] = 1
    if lane[nextlast] == 1 and lane[last] + lane[0] == 0:
        new_lane[nextlast] = 0
        new_lane[last] = 1
        cars_moved[nextlast] = 1
    if lane[last] == 1 and lane[0] + lane[1] == 0:
        new_lane[last] = 0
        new_lane[0] = 1
        cars_moved[last] = 1
    return lane, cars_moved

def run184(steps, lane):
    speeds = []

    n_cars=np.sum(lane)
    for step_num in range(steps):
        lane, cars_moved = step184(lane)
        
        speed = np.sum(cars_moved) / n_cars   # Calculate speed as the fraction of cars that moved
        speeds.append(speed)

    return lane, speeds

def run58336(steps, lane):
    speeds = []

    n_cars=np.sum(lane)
    for step_num in range(steps):
        lane, cars_moved = step58336(lane)
        
        speed = np.sum(cars_moved) / n_cars   # Calculate speed as the fraction of cars that moved
        speeds.append(speed)

    return lane, speeds