## Imports
import charges
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import seaborn as sns
import pandas as pd
## Functions

def plot_points(radius, points, ch, force):
    theta = np.linspace(0, 2 * np.pi, 150)
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)
    
    fig, ax = plt.subplots(1)
    ax.plot(a, b, color='black')
    scatter, = ax.plot(points[:,0], points[:,1], 'bo')
    ax.set_aspect(1)

    if force:
        for p in range(ch.n_particles):
            x,y = ch.particles[p]
            force_x, force_y = ch.total_force_on_particle(p)
            plt.quiver(x, y, force_x, force_y)

    # def animate(frame_num):
    #     for p in range(ch.n_particles):
    #         ch.move_particle_random(p, 0.1)
    #     data = ch.particles
    #     scatter.set_xdata(data[:,0])
    #     scatter.set_ydata(data[:,1])
    #     return scatter
    
    # anim = FuncAnimation(fig, animate, frames=100, interval=5)
    plt.show()

    # return anim


def plot_convergence(fname, pic_name, first_n_iters=None):
    res_df = pd.read_csv(f"logged_data/{fname}.csv", skiprows=[1]).rename(columns={"Unnamed: 0": "Iterations"})
    if first_n_iters:
        res_df = res_df[:first_n_iters]

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
    ax1.scatter(x_iters[::-1], stats['mean'], color='black', s=20)

    sns.set_theme(style="white")
    ax2 = ax1.twinx()
    sns.lineplot(ax=ax2, x=res_df["Iterations"], y=res_df["Temperatures"], color='red')

    ax1.set_ylabel("Potential Energie, E")
    ax1.set_xlabel("Iterations")
    plt.xlim((1, 3000))
    plt.savefig(pic_name, dpi=300, bbox_inches='tight')
