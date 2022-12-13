## Imports
import numpy as np
import numba as nb
import math
from itertools import combinations
import os
import csv
from pathlib import Path

import pandas
import pandas as pd

# Class definition

class Charges():
    """ Class that handles charge particles and the simulated annealing algorithm
    """

    def __init__(self, n_particles, radius=1, step_size=0.01):
        self.n_particles = n_particles
        self.radius = radius
        self.particles = self.generate_points(radius)
        self.pot_energy = self.evaluate_configuration()
        self.step_size = step_size

    def generate_points(self, radius=1):
        """ Generate n_particles within a circle 
        """
        rng = np.random.default_rng(None)
        points = []
        for n in range(self.n_particles):
            # random angle and distance from center
            alpha = 2 * np.pi * rng.random()
            r = radius * np.sqrt(rng.random())
            # coordinates
            x = r * np.cos(alpha)
            y = r * np.sin(alpha)
            points.append([x, y])

        return np.array(points)

    def evaluate_configuration(self):
        """ Calculate the total energy of the current configuration
        """
        total = 0
        for i, j in list(combinations(range(self.n_particles), 2)):
            p1, p2 = self.particles[i], self.particles[j]
            total += 1 / self.euclidean(p1, p2)

        return total

    @staticmethod
    def euclidean(p1, p2):
        """ Compute the euclidean distance between two points
        """
        return np.linalg.norm(p2 - p1)

    @staticmethod
    def euclidean_vec(combs):
        """ Compute the euclidean distance between two points for a
         np array of all point combinations: 4 columns with n rows.
        """

        return np.sqrt((combs[:, 0] - combs[:, 2]) ** 2 +
                       (combs[:, 1] - combs[:, 3]) ** 2)
        # return np.linalg.norm(p2 - p1)

    def evaluate_configuration_fast(self):
        """No idea if this is acc faster."""

        # generate combinations with np broadcasting
        m, n = self.particles.shape
        comb = np.zeros((m, m, n + n), dtype=float)
        comb[:, :, :n] = self.particles[:, None, :]
        comb[:, :, n:] = self.particles
        comb.shape = (m * m, -1)  # shape is 4 columns w len(particles) rows

        # now we also get p1 -p1 combinations, but
        # their inter-particle distance is 0, so we can ignore that fact
        dists = self.euclidean_vec(comb)
        return 1 / np.sum(dists[dists != 0])

    def check_in_circle(self, p):
        """ Check if point p is within the circle
            returns 0 if it is outside, 1 if it is inside
        """
        distance = self.euclidean([0, 0], p)
        if distance > self.radius:
            ## Maybe let it bounce
            return 0
        else:
            return 1

    def move_particle_random(self, p):
        """ Tries to move particle p (index for particles) by stepsize
            returns 1 if success, 0 if failed
        """
        step = self.generate_points(1)[0]
        particle = self.particles[p]
        if self.check_in_circle(particle + step):
            self.particles[p] = particle + step
            return 1
        else:
            return 0

    def move_particle_random_new(self, p):
        rng = np.random.default_rng(None)
        delta = rng.uniform(-self.step_size, self.step_size, 2)
        new_loc = self.particles[p, :] + delta
        if self.check_in_circle(new_loc):
            self.particles[p, :] = new_loc
            return 1
        else:
            return 0

    def do_SA_step(self, p, cur_temp, single_rand_particle):
        """
        Does a single SA step:
        1. Move point randomly
        2. Evaluate configuration
        3. If better accept new config
        4. If worse accept depending on temperature
        p = index of particle
        """
        rng = np.random.default_rng(None)
        if single_rand_particle:
            p = rng.integers(0, len(self.particles))
        # save olf state of the system
        last_particles = np.copy(self.particles)

        # do SA move
        self.move_particle_random(p)
        # Evaluate new configuration
        new_pot_energy = self.evaluate_configuration()
        # accept if better independent of temp
        if new_pot_energy <= self.pot_energy:
            self.pot_energy = new_pot_energy
        # accept move is chance of acceptance is greater based on current temperature
        elif np.exp((self.pot_energy - new_pot_energy) / cur_temp) >= rng.random():
            self.pot_energy = new_pot_energy
        # reject move
        else:
            self.particles = last_particles

    @staticmethod
    def generate_temperature_list(low_temp, high_temp, n_temps, schedule):
        """
        Schedule: Linear, exponential, geometric
        """
        if schedule == "linear":
            return np.linspace(high_temp, low_temp, n_temps)
        elif schedule == "exponential_even_spacing":
            return np.geomspace(high_temp, low_temp, n_temps)
        elif schedule == "exponential_0.003":
            return np.exp(-np.arange(0, n_temps) * 30 / n_temps) * high_temp
        else:
            raise TypeError("%s is not a valid cooling schedule."
                            " Try linear, exponential_even_spacing or exponential_0.003" % schedule)

    def write_data(self, schedule,all_temps, chain_length, all_energies):
        fname = f"{len(self.particles)}_{schedule}_{len(all_temps)}_{chain_length}"
        if not os.path.exists('logged_data'):
            os.makedirs("logged_data")
        my_file = Path(os.path.join("logged_data", fname + ".csv"))

        if not my_file.is_file():
            with open(os.path.join("logged_data", fname + ".csv"), "w+") as f:
                wr = csv.writer(f)
                wr.writerow(["all_energies"])

        write_df = pandas.DataFrame(columns=['Temperatures', 'Chain_indices', 'Potential_energy'])
        list_temperatures = np.repeat(all_temps, chain_length*self.n_particles)
        chain_indices = np.repeat(np.arange(0, chain_length), len(all_temps)*self.n_particles)
        write_df["Temperatures"] = list_temperatures
        write_df["Chain_indices"] = chain_indices
        write_df["Potential_energy"] = all_energies

        write_df.to_csv(os.path.join("logged_data", fname + ".csv"))

        # log the data
        # with open(os.path.join("logged_data", fname + ".csv"), "a") as f:
        #     wr = csv.writer(f)
        #     wr.writerow(all_energies)
        #     wr.writerow(list_temperatures)
        #     wr.writerow(chain_indices)

    def iterate_SA_optimize(self, low_temp, high_temp, n_temps, schedule, chain_length, single_rand_particle=False):

        # save potential energy for each iteration
        nr_points = len(self.particles)
        if single_rand_particle:
            nr_points = 1
        if low_temp == 0:
            low_temp += 0.01

        all_temps = self.generate_temperature_list(low_temp, high_temp,
                                                   n_temps, schedule)

        all_energies = np.empty(n_temps * nr_points* chain_length)
        p_idx = 0
        for cur_temp in all_temps:
            for chain_index in range(chain_length):
                for p in range(nr_points):
                    all_energies[p_idx] = self.pot_energy
                    self.do_SA_step(p, cur_temp, single_rand_particle)
                    p_idx += 1

        self.write_data(schedule, all_temps, chain_length, all_energies)
        return self.particles


    def total_force_on_particle(self, p):
        F = np.zeros(2)
        for particle in range(self.n_particles):
            if particle != p:
                dir = self.particles[p] - self.particles[particle]
                dist = self.euclidean(self.particles[particle], self.particles[p])
                F += dir / (dist ** 3)
        return F / (self.n_particles - 1)