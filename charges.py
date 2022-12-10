## Imports
import numpy as np
import numba as nb
import math
from itertools import combinations
import os
import csv
from pathlib import Path


# Class definition

class Charges():
    """ Class that handles charge particles and the simulated annealing algorithm
    """

    def __init__(self, n_particles, radius, step_size=0.01):
        self.n_particles = n_particles
        self.radius = radius
        self.particles = self.generate_points(n_particles, radius)
        self.pot_energy = self.evaluate_configuration()
        self.step_size = step_size

    def generate_points(self, n_particles, radius, seed=None):
        """ Generate n_particles within a circle 
        """
        rng = np.random.default_rng(seed)
        points = []
        for n in range(n_particles):
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

    def evaluate_configuration_fast(self):
        """No idea if this is acc faster."""

        # generate combinations
        m, n = self.particles.shape
        comb = np.zeros((m, m, n + n), dtype=int)
        comb[:, :, :n] = self.particles[:, None, :]
        comb[:, :, n:] = self.particles
        comb.shape = (m * m, -1)  # shape is 4 columns w len(particles) rows
        # now we also get p1 -p1 combinations
        res = 1 / self.euclidean(comb[:, :2], comb[:, 2:])
        return np.sum(res)

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

        return np.sqrt((combs[:, 0] - combs[:, 1]) ** 2 +
                       (combs[:, 2] - combs[:, 3]) ** 2)
        # return np.linalg.norm(p2 - p1)

    def evaluate_configuration_fast(self):
        """No idea if this is acc faster."""

        # generate combinations with np broadcasting
        m, n = self.particles.shape
        comb = np.zeros((m, m, n + n), dtype=int)
        comb[:, :, :n] = self.particles[:, None, :]
        comb[:, :, n:] = self.particles
        comb.shape = (m * m, -1)  # shape is 4 columns w len(particles) rows

        # now we also get p1 -p1 combinations, but
        # their inter-particle distance is 0, so we can ignore that fact

        return 1 / np.sum(self.euclidean_vec(comb))

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
        step = self.generate_points(1, self.step_size)[0]
        particle = self.particles[p]
        if self.check_in_circle(particle + step):
            self.particles[p] = particle + step
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
            p = rng.integers(0, len(self.particles) + 1)
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
        elif schedule == "exponential_0.01":
            return np.exp(-np.arange(low_temp, high_temp) * 0.001) * high_temp
        else:
            raise TypeError("%s is not a valid cooling schedule."
                            " Try linear, exponential_even_spacing or exponential_0.01" % schedule)

    def write_data(self, schedule, iterations, all_energies):
        fname = f"{len(self.particles)}_{schedule}_{iterations}"
        if not os.path.exists('logged_data'):
            os.makedirs("logged_data")
        my_file = os.Path(os.path.join("logged_data", fname + ".csv"))

        if not my_file.is_file():
            with open(os.path.join("logged_data", fname + ".csv"), "w+") as f:
                wr = csv.writer(f)
                wr.writerow(["all_energies", "_", "_", "_"])

        # log the data
        with open(os.path.join("logged_data", fname + ".csv"), "a") as f:
            wr = csv.writer(f)
            wr.writerow(all_energies)
            # wr.writerow([_, _, _, _, _])

    def iterate_SA_optimize(self, low_temp, high_temp, iterations, schedule, single_rand_particle=False):

        # save potential energy for each iteration
        nr_points = len(self.particles)
        if single_rand_particle:
            nr_points = 1

        all_temps = self.generate_temperature_list(low_temp, high_temp, iterations, schedule)
        all_energies = np.empty(iterations * nr_points)
        p_idx = 0
        for cur_temp in all_temps:
            for p in range(nr_points):
                all_energies[p_idx] = self.pot_energy
                self.do_SA_step(p, cur_temp, single_rand_particle)
                p_idx += 1

        self.write_data(schedule, iterations, all_energies)
        return self.particles

    def total_force_on_particle(self, p):
        """
        Returns total force and its direction
        p is particle index
        """

        ##### NOT FINISHED YET #######
        # 1. write this so it only iterates over single_particle x all_others
        # 2. Make sure it returns something sensible
        # 3. Make sure it acc works

        # generate combinations with np broadcasting
        force_f = lambda rij: rij / np.abs(rij ** 3)
        m, n = self.particles.shape
        comb = np.zeros((m, m, n + n), dtype=int)
        comb[:, :, :n] = self.particles[:, None, :]
        comb[:, :, n:] = self.particles
        comb.shape = (m * m, -1)

        deltas_x = np.abs(comb[0] - comb[2])
        deltas_y = np.abs(comb[1] - comb[3])

        # find force vector
        # https://www.kristakingmath.com/blog/magnitude-and-angle-of-the-resultant-force
        # https://www.dummies.com/article/academics-the-arts/science/physics/how-to-find-the-angle-and-magnitude-of-a-vector-173966/
        # WTF am i doing
        force_dirs = np.radians(-np.degrees(np.arctan(deltas_y / deltas_x))) + np.pi / 2
        forces = force_f(self.euclidean_vec(comb))
        res = np.vstack(np.sin(force_dirs), np.cos(force_dirs)).T * forces
        return res, force_dirs, forces
