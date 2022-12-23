import charges
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np
import pandas as pd
from IPython import display

import seaborn as sns

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

def plot_convergence_v2(fname,pic_name,first_n_iters=None, plot_points = True, plot_original=False):
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
