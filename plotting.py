## Imports
import charges
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

# Global vars
p_idx = 0
temp_index = 0

## Functions

def plot_points(points, charge=None, radius=1.0, force=False):
    """ Plot a single configuration of the particles, if force is set to 
        true also displays the vectors of the forces, black individual green 
        the normalized sum
    """
    theta = np.linspace(0, 2 * np.pi, 150)
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)
    
    fig, ax = plt.subplots(1)
    ax.plot(a, b, color='black')
    ax.plot(points[:,0], points[:,1], 'bo')
    ax.set_aspect(1)

    if force:
        for p in range(charge.n_particles):
            forces = charge.all_forces_on_particle(p)
            x,y = charge.particles[p]

            for f in forces:
                plt.quiver(x, y, f[0], f[1], angles='xy', scale_units='xy', scale=1)

            force_x, force_y = charge.total_force_on_particle(p)
            plt.quiver(x, y, force_x, force_y, angles='xy', scale_units='xy', scale=1, color='green')
    
    plt.show()

def animate_convergence(ch, low_temp, high_temp, n_temps, schedule, chain_length, force=False):
    """ Experimental function """
    theta = np.linspace(0, 2 * np.pi, 150)
    a = ch.radius * np.cos(theta)
    b = ch.radius * np.sin(theta)
    
    fig, ax = plt.subplots(1)
    ax.plot(a, b, color='black')
    scatter, = ax.plot(ch.particles[:,0], ch.particles[:,1], 'bo')
    ax.set_aspect(1)
    ax.set_title(f"Total energy of the system: {ch.evaluate_configuration()}")

    all_temps = ch.generate_temperature_list(low_temp, high_temp,
                                                   n_temps, schedule)

    all_energies = np.empty(n_temps * ch.n_particles * chain_length)

    def animate(frame_num):
        global temp_index
        global p_idx
        cur_temp = all_temps[temp_index]
        print(f"Iteration {temp_index}/{n_temps} at {cur_temp} degrees", end='\r', flush=True)
        temp_index += 1
        for chain_index in range(chain_length):
            for p in range(ch.n_particles):
                all_energies[p_idx] = ch.pot_energy
                ch.do_SA_step(p, cur_temp, force)
                p_idx += 1
        
        data = ch.particles
        scatter.set_xdata(data[:,0])
        scatter.set_ydata(data[:,1])
        return scatter
    
    anim = FuncAnimation(fig, animate, frames=n_temps, interval=0)
    plt.show()
    # return anim

def plot_convergence(filename):
    fig, ax = plt.subplots()
    data = pd.read_csv(filename)
    energy = data["Potential_energy"]
    temp = data["Temperatures"]

    ax.plot(energy)
    ax.set_ylabel('Total energy of the system', color='blue')
    ax2 = plt.twinx(ax)
    ax2.plot(temp, color='red')
    ax2.set_ylabel('Temperature', color='red')
    plt.show()
