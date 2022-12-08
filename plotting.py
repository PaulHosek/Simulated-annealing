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
    # plotpoints = []
    # for p in points:
    scatter = ax.scatter(points[:,0], points[:,1], color='blue')
    ax.set_aspect(1)
    # plt.show()

    return fig, ax, scatter, 