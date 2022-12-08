## Imports
import matplotlib.pyplot as plt
import numpy as np

## Functions

def plot_points(radius, points):
    theta = np.linspace(0, 2 * np.pi, 150)
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)
    
    fig, ax = plt.subplots(1)
    ax.plot(a, b, color='black')
    for p in points:
        ax.scatter(p[0], p[1], color='blue')
    ax.set_aspect(1)
    plt.show()