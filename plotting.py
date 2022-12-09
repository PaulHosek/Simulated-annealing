## Imports
import charges
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

## Functions

def plot_points(radius, points, ch):
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

    def animate(frame_num):
        for p in range(ch.n_particles):
            ch.move_particle_random(p, 0.1)
        # test.append(ch.particles)
        scatter.set_offsets(ch.particles[:,0], ch.particles[:,1])
        return scatter

    anim = FuncAnimation(fig, animate, frames=100, interval=5)
    plt.show()

    # return fig, ax, scatter, 