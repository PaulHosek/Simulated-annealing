## Imports
import charges
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np
import pandas as pd
from IPython import display

import seaborn as sns
# Global vars
p_idx = 0
temp_index = 1

## Functions

def plot_points(points, charge=None, radius=1.0, force=False, show=True):
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
        ax.set_ylim(-2,2)
        ax.set_xlim(-2,2)
        for p in range(charge.n_particles):
            forces = charge.all_forces_on_particle(p)
            x,y = charge.particles[p]

            for f in forces:
                plt.quiver(x, y, f[0], f[1], alpha=0.2, angles='xy', scale_units='xy', scale=1)

            force_x, force_y = charge.total_force_on_particle(p)
            plt.quiver(x, y, force_x, force_y, angles='xy', scale_units='xy', scale=1, color='green')
    if show:
        plt.show()

def animate_convergence(ch, low_temp, high_temp, n_temps, schedule, chain_length, wavy=False, force=False):
    """ Experimental function """
    forcelist = np.append(np.zeros(int(n_temps*0.75)), (np.ones(int(n_temps*0.25))))
    theta = np.linspace(0, 2 * np.pi, 150)
    a = ch.radius * np.cos(theta)
    b = ch.radius * np.sin(theta)
    
    fig, ax = plt.subplots()
    ax.plot(a, b, color='black')
    scatter, = ax.plot(ch.particles[:,0], ch.particles[:,1], 'bo')
    ax.set_aspect(1)

    all_temps = ch.generate_temperature_list(low_temp, high_temp,
                                                   n_temps, schedule, wavy=wavy)

    all_energies = np.empty(n_temps * ch.n_particles * chain_length)

    def animate(frame_num):
        cur_temp = all_temps[frame_num]
        print(f"Iteration {frame_num+1}/{n_temps} at {cur_temp} degrees, energy: {ch.evaluate_configuration()}", end='\r', flush=True)
            
        for chain_index in range(chain_length):
            for p in range(ch.n_particles):
                ch.do_SA_step(p, cur_temp, forcelist[frame_num])
        
        data = ch.particles
        scatter.set_xdata(data[:,0])
        scatter.set_ydata(data[:,1])
        return scatter
    
    anim = FuncAnimation(fig, animate, frames=n_temps, interval=50, repeat=False)
    return anim

def plot_convergence_v1(filename):
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

def plot_convergence(fname,pic_name,first_n_iters=None, plot_points = True, plot_original=False):
    res_df = pd.read_csv(f"logged_data/{fname}.csv", skiprows=[1]).rename(columns={"Unnamed: 0": "Iterations"})




    # calculate mean and 95% ci for temperature level
    stats = res_df.groupby(['Temperatures']).agg(['mean', 'sem'])
    x_iters = stats["Iterations"]['mean']
    stats = stats["Potential_energy"]
    stats['ci95_hi'] = stats['mean'] + 1.96 * stats['sem']
    stats['ci95_lo'] = stats['mean'] - 1.96 * stats['sem']
    stats = stats.iloc[::-1]

    # draw
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    ax1 = sns.lineplot(x=x_iters, y=stats['mean'], sort=False, color='blue', label='Mean Energy in Chain')
    sns.lineplot(ax=ax1, x=x_iters, y=stats['ci95_hi'], sort=False, color='grey', linestyle='--', label='95% CI')
    sns.lineplot(ax=ax1, x=x_iters, y=stats['ci95_lo'], sort=False, color='grey', linestyle='--')
    ax1.fill_between(x_iters[::-1], stats['mean'], stats['ci95_hi'], color='lightblue', alpha=0.5)
    ax1.fill_between(x_iters[::-1], stats['mean'], stats['ci95_lo'], color='lightblue', alpha=0.5)
    if plot_points:
        ax1.scatter(x_iters[::-1], stats['mean'], color='black', s=20)

    sns.set_theme(style="white")
    ax2 = ax1.twinx()
    sns.lineplot(ax=ax2, x=res_df["Iterations"], y=res_df["Temperatures"], color='red')

    if plot_original:
        sns.lineplot(ax=ax1, x=res_df["Iterations"], y=res_df["Potential_energy"], color='black',alpha=0.2)
    ax1.legend(loc='lower left')
    ax1.set_ylabel("Potential Energy, E")
    ax1.set_xlabel("Evaluations")
    if first_n_iters:
        plt.xlim((1, first_n_iters))
    plt.savefig('Images/'+pic_name+".svg",dpi=300,bbox_inches='tight')
