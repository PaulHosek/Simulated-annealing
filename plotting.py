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
    # forcelist = np.append(np.zeros(int(n_temps*0.75)), (np.ones(int(n_temps*0.25))))
    forcelist = np.tile(np.append(np.zeros(int(chain_length/2)), np.ones(int(chain_length/2))), n_temps)
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


def plot_convergence(fname,pic_name=None, first_n_iters=None, plot_points = True, plot_raw_data=False, plot_final_state = True):
    """
    Plot convergence, temperature and final configuration in single plot and save as svg.
    @param fname: file name for the energies
    @param pic_name: the name of the output image, if black will use "final_particles_" + fname + '.csv'
    @param first_n_iters: only show the first n evaluations
    @param plot_points: if plot the mean energy as additional scatterpoints over the curve
    @param plot_raw_data: if plot the raw data under the plot
    """
    if not pic_name:
        pic_name = fname
    particles_fname = "final_particles_" + fname + '.csv'
    final_config = np.loadtxt('logged_data/' + particles_fname, delimiter=',')
    res_df = pd.read_csv(f"logged_data/{fname}.csv", skiprows=[1]).rename(columns={"Unnamed: 0": "Iterations"})


    # plotting.plot_points(my_charge.particles)
    # ins.plotting.plot_points(my_charge.particles)

    def insert_plot(points,ax, radius=1.0):
        """ Plot the final configuration
        """
        theta = np.linspace(0, 2 * np.pi, 150)
        a = radius * np.cos(theta)
        b = radius * np.sin(theta)
        # ax.axis('off')
        ax.set_xticks([],[])
        ax.set_yticks([],[])
        ax.plot(a, b, color='black')
        ax.scatter(points[:,0], points[:,1],color='blue',marker='o')
        ax.set_aspect(1)
        ax.axhline(0,.05,1-0.05,color='black')
        ax.axvline(0,.05,1-0.05,color='black')



    # calculate mean and 95% ci for temperature level
    stats = res_df.groupby(['Temperatures']).agg(['mean', 'sem'])
    x_iters = stats["Iterations"]['mean']
    stats = stats["Potential_energy"]
    stats['ci95_hi'] = stats['mean'] + 1.96 * stats['sem']
    stats['ci95_lo'] = stats['mean'] - 1.96 * stats['sem']
    stats = stats.iloc[::-1]

    # draw
    fax1 = plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    sns.set_context("notebook", font_scale=1.5)

    # draw main convergence data, CI and means
    ax1 = sns.lineplot(x=x_iters, y=stats['mean'], sort=False, color='blue', label='Mean Energy')
    # plot raw data
    if plot_raw_data:
        sns.lineplot(ax=ax1, x=res_df["Iterations"], y=res_df["Potential_energy"],
                     color='black',alpha=0.15,label='Raw energy')
    sns.lineplot(ax=ax1, x=x_iters, y=stats['ci95_hi'], sort=False, color='cornflowerblue', linestyle='--', label='95% CI')
    sns.lineplot(ax=ax1, x=x_iters, y=stats['ci95_lo'], sort=False, color='cornflowerblue', linestyle='--')
    ax1.fill_between(x_iters[::-1], stats['mean'], stats['ci95_hi'], color='cornflowerblue', alpha=0.5)
    ax1.fill_between(x_iters[::-1], stats['mean'], stats['ci95_lo'], color='cornflowerblue', alpha=0.5)
    if plot_points:
        ax1.scatter(x_iters[::-1], stats['mean'], color='black', s=20)

    # add temperature plot
    sns.set_theme(style="white")
    ax2 = ax1.twinx()
    temp_line = sns.lineplot(ax=ax2, x=res_df["Iterations"], y=res_df["Temperatures"], color='red', label='Temperature')

    # insert final configuration
    if plot_final_state:
        ins = ax1.inset_axes([0.65,0.46,0.3,0.3*1.5])
        insert_plot(final_config,ax=ins)



    # legend
    ax2.legend([],[],frameon=False)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(lines + lines2, labels + labels2, loc='upper center', framealpha=1, prop={'size': 14})
    leg.get_frame().set_edgecolor('black')

    # color the axis
    ax2.spines['right'].set_color('red')
    ax1.spines['left'].set_color('blue')
    ax2.spines['right'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)


    ax1.set_ylabel(r"Potential Energy, $E$")
    ax1.set_xlabel("Evaluations")
    temp_line.set_ylabel("Temperature", fontsize = 18)
    if first_n_iters:
        plt.xlim((1, first_n_iters))
    plt.savefig('Images/'+pic_name+".svg",dpi=300,bbox_inches='tight')

def plot_convergence_force(fname1, fname2, fname3,schedule, names=('no force','full force', "late force"),
                           pic_name='no_force', first_n_iters=None, plot_points=True,plot_raw_data=False):
    """
    Compare no force, full force and late force in single plot
    @param fname1,fname2,fname3: file name for the energies for the 3 force variants
    @param pic_name: the name of the output image, if black will use "final_particles_" + fname + '.csv'
    @param first_n_iters: only show the first n evaluations
    @param plot_points: if plot the mean energy as additional scatterpoints over the curve
    """
    # draw
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    sns.set_context("notebook", font_scale=1.5)

    colors = ['red','blue','green']
    region_colors = ['indianred','cornflowerblue','mediumseagreen']

    for idx, fname in enumerate([fname1, fname2, fname3]):

        # get data
        particles_fname = "final_particles_" + fname + '.csv'
        final_config = np.loadtxt('logged_data/' + particles_fname, delimiter=',')
        res_df = pd.read_csv(f"logged_data/{fname}.csv", skiprows=[1]).rename(columns={"Unnamed: 0": "Iterations"})

        # calculate mean and 95% ci for temperature level
        stats = res_df.groupby(['Temperatures']).agg(['mean', 'sem'])
        x_iters = stats["Iterations"]['mean']
        stats = stats["Potential_energy"]
        stats['ci95_hi'] = stats['mean'] + 1.96 * stats['sem']
        stats['ci95_lo'] = stats['mean'] - 1.96 * stats['sem']
        stats = stats.iloc[::-1]

        # draw main convergence data, CI and means
        ax1 = sns.lineplot(x=x_iters, y=stats['mean'], sort=False, color=colors[idx], label=names[idx])
        # plot raw data
        if plot_raw_data:
            sns.lineplot(ax=ax1, x=res_df["Iterations"], y=res_df["Potential_energy"],
                         color=colors[idx], alpha=0.15, label='Raw energy')
        sns.lineplot(ax=ax1, x=x_iters, y=stats['ci95_hi'], sort=False, color=region_colors[idx], linestyle='--')
        sns.lineplot(ax=ax1, x=x_iters, y=stats['ci95_lo'], sort=False, color=region_colors[idx], linestyle='--')
        ax1.fill_between(x_iters[::-1], stats['mean'], stats['ci95_hi'], color=region_colors[idx], alpha=0.5)
        ax1.fill_between(x_iters[::-1], stats['mean'], stats['ci95_lo'], color=region_colors[idx], alpha=0.5)
        if plot_points:
            ax1.scatter(x_iters[::-1], stats['mean'], color='black', s=20)

        # add temperature plot
    lines, labels = ax1.get_legend_handles_labels()
    sns.set_theme(style="white")
    ax2 = ax1.twinx()
    temp_line = sns.lineplot(ax=ax2, x=res_df["Iterations"], y=res_df["Temperatures"], color='black',linestyle=':',
                                 label='Temperature',legend=False)


    leg = ax1.legend(lines, labels, loc='upper right', framealpha=1, prop={'size': 14})
    leg.get_frame().set_edgecolor('black')



    ax1.set_ylabel(r"Potential Energy, $E$")
    ax1.set_xlabel("Evaluations")

    if first_n_iters:
        plt.xlim((1, first_n_iters))
    plt.text(ax1.get_xlim()[1]*0.5, ax1.get_ylim()[1]*0.95, f'Cooling schedule: {schedule}', style='italic',
            bbox={'facecolor':'white','edgecolor': 'black', 'alpha': 1, 'pad': 10}
             ,horizontalalignment='center', verticalalignment='center')

    plt.savefig('Images/' + pic_name + ".svg", dpi=300, bbox_inches='tight')





def plot_convergence_only_raw(fname1, fname2, fname3,schedule, names=('no force','full force', "late force"),
                           pic_name='no_force', first_n_iters=None,
                           plot_temp = False):
    """
    Compare no force, full force and late force in single plot
    @param fname1,fname2,fname3: file name for the energies for the 3 force variants
    @param pic_name: the name of the output image, if black will use "final_particles_" + fname + '.csv'
    @param first_n_iters: only show the first n evaluations
    @param plot_points: if plot the mean energy as additional scatterpoints over the curve
    """
    # draw
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    sns.set_context("notebook", font_scale=1.5)

    colors = ['red','blue','green']

    for idx, fname in enumerate([fname1, fname2, fname3]):

        # get data
        particles_fname = "final_particles_" + fname + '.csv'
        final_config = np.loadtxt('logged_data/' + particles_fname, delimiter=',')
        res_df = pd.read_csv(f"logged_data/{fname}.csv", skiprows=[1]).rename(columns={"Unnamed: 0": "Iterations"})

        # draw main convergence data, CI and means
        # plot raw data
        ax1 = sns.lineplot(x=res_df["Iterations"], y=res_df["Potential_energy"],
                     color=colors[idx], alpha=0.8, label=names[idx])



        # add temperature plot
    lines, labels = ax1.get_legend_handles_labels()
    if plot_temp:
        sns.set_theme(style="white")
        ax2 = ax1.twinx()
        temp_line = sns.lineplot(ax=ax2, x=res_df["Iterations"], y=res_df["Temperatures"], color='black',linestyle=':',
                                 label='Temperature',legend=False, linewidth='3')
        temp_line.set_ylabel("Temperature", fontsize=18)
        # color the axis
        # ax2.spines['right'].set_color('red')
        # ax1.spines['left'].set_color('blue')
        # ax2.spines['right'].set_linewidth(2)
        # ax1.spines['left'].set_linewidth(2)
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
        ax1.set_ylabel("Temperature")


    leg = ax1.legend(lines, labels, loc='upper right', framealpha=1, prop={'size': 14})
    leg.get_frame().set_edgecolor('black')



    ax1.set_ylabel(r"Potential Energy, $E$")
    ax1.set_xlabel("Evaluations")

    if first_n_iters:
        plt.xlim((1, first_n_iters))
    plt.text(ax1.get_xlim()[1]*0.5, ax1.get_ylim()[1]*0.95, f'Cooling schedule: {schedule}', style='italic',
            bbox={'facecolor':'white','edgecolor': 'black', 'alpha': 1, 'pad': 10}
             ,horizontalalignment='center', verticalalignment='center')

    plt.savefig('Images/' + pic_name + ".svg", dpi=300, bbox_inches='tight')