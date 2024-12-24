import numpy as np
import queue0 as jamq
import matplotlib.pyplot as plt

L = 100
car_pos = np.empty([0], dtype=int)  # Ensure this is an integer array
waiting = 0
delta = 2
critical_rate = 1/(delta+1)
arrival_rate = critical_rate*(1-1e-3)

for t in range(int(2*L/critical_rate)):
    car_pos, waiting = jamq.step(L, car_pos, waiting, arrival_rate, delta)  # Corrected order of arguments

plt.figure(figsize=(8, 12))
for t in range(10000):
    car_pos, waiting = jamq.step(L, car_pos, waiting, arrival_rate, delta)  # Corrected order of arguments
    #print(car_pos)
    plt.subplot(2, 1, 1)
    plt.plot(t+1, len(car_pos)/L, color='b', marker='o', linestyle='')
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(t+1, waiting/L, color='r', marker='o', linestyle='')
    plt.grid()
    
plt.tight_layout()
plt.show()

    

    