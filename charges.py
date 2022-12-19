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

    def move_particle_force(self, p):
        """ Tries to move particle p (index for particles) by force
            returns 1 if success, 0 if failed
        """
        F = self.total_force_on_particle(p)
        step = self.step_size * F
        particle = self.particles[p]
        if self.check_in_circle(particle + step):
            self.particles[p] = particle + step
        else:
            self.move_along_edge(p, particle, F)
            # return 1

    def move_along_edge(self, p, particle, force):
        """ Move particle along edge of the circle to avoid it getting stuck 
            in local minimum
        """
        # F = np.linalg.norm(force)
        coord = particle + force

        theta_force = np.arctan2(coord[1], coord[0])
        theta_part = np.arctan2(particle[1], particle[0])

        # diff = np.abs(theta_force - theta_part)
        bounce = self.radius #* (1-diff) #* F
        step = np.pi / 180

        if np.sign(theta_part) == -1.0:
            if theta_force > theta_part:
                theta = theta_part + step
            else:
                theta = theta_part - step
        else:
            if theta_force > theta_part:
                theta = theta_part + step
            else:
                theta = theta_part - step
        new = np.array([bounce * np.cos(theta), bounce * np.sin(theta)])
        self.particles[p] = new

    def move_particle_random_new(self, p):
        rng = np.random.default_rng(None)
        delta = rng.uniform(-self.step_size, self.step_size, 2)
        new_loc = self.particles[p, :] + delta
        if self.check_in_circle(new_loc):
            self.particles[p, :] = new_loc
            return 1
        else:
            return 0

    def do_SA_step(self, p, cur_temp, force):
        """
        Does a single SA step:
        1. Move point randomly
        2. Evaluate configuration
        3. If better accept new config
        4. If worse accept depending on temperature
        p = index of particle
        """
        rng = np.random.default_rng(None)
        # save old state of the system
        last_particles = np.copy(self.particles)

        # do SA move
        if force:
            self.move_particle_force(p)
        else:
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
    def generate_temperature_list(low_temp, high_temp, n_temps, schedule, wavy = False):
        """
        Schedule: Linear, evenly spaced exponential, exponential 0.003.
        If wavy == True, transform function into wavy variant.
        """
        lb_f = lambda x: np.where(x < low_temp, low_temp, x)
        if wavy:
            wave_transf = lambda x: x + np.exp(-0.001 * -x) * np.sin(1 * -x)
            wave_func = lambda x: lb_f(wave_transf(x))
        else:
            wave_func = lambda x: lb_f(x)
        if schedule == "linear":
            return wave_func(np.linspace(high_temp, low_temp, n_temps))
        elif schedule == "exponential_even_spacing":
            return wave_func(np.geomspace(high_temp, low_temp, n_temps))
        elif schedule == "exponential_0.003":
            return wave_func(np.exp(-np.arange(0, n_temps) * 30 / n_temps) * high_temp)
        else:
            raise TypeError("%s is not a valid cooling schedule."
                            " Try linear, exponential_even_spacing or exponential_0.003" % schedule)

    def write_data(self, schedule, all_temps, chain_length, all_energies, force, wavy):
        force_nam = ['noforce', 'force', 'lateforce', 'halfforce'][force]
        wavy_nam = 'wavy' if wavy else 'nowavy'
        fname = f"{len(self.particles)}_{schedule}_{len(all_temps)}_{chain_length}_{force_nam}_{wavy_nam}"
        if not os.path.exists('logged_data'):
            os.makedirs("logged_data")
        my_file = Path(os.path.join("logged_data", fname + ".csv"))

        if not my_file.is_file():
            with open(os.path.join("logged_data", fname + ".csv"), "w+") as f:
                wr = csv.writer(f)
                wr.writerow(["all_energies"])

        write_df = pandas.DataFrame(columns=['Temperatures', 'Chain_indices', 'Potential_energy'])
        list_temperatures = np.repeat(all_temps, chain_length*self.n_particles)
        chain_indices = np.tile(np.arange(0, chain_length), len(all_temps)*self.n_particles)
        write_df["Temperatures"] = list_temperatures
        write_df["Chain_indices"] = chain_indices
        write_df["Potential_energy"] = all_energies

        write_df.to_csv(os.path.join("logged_data", fname + ".csv"))

        # write final particle configuration
        final_config = self.particles
        particles_fname = "final_particles_"+fname + ".csv"
        print(particles_fname)
        np.savetxt('logged_data/'+particles_fname, final_config, delimiter=",")


    def iterate_SA_optimize(self, low_temp, high_temp, n_temps, schedule, chain_length, force=0, wavy=False):
        nr_eval = n_temps * self.n_particles * chain_length
        if force == 0:
            forcelist = np.zeros(nr_eval)
        elif force == 1:
            forcelist = np.ones(nr_eval)
        elif force == 2:
            forcelist = np.append(np.zeros(int(nr_eval*0.75)), (np.ones(int(nr_eval*0.25))))
        else:
            forcelist = np.tile(np.append(np.zeros(int(chain_length/2)), np.ones(int(chain_length/2))), nr_eval)


        # save potential energy for each iteration
        
        if low_temp == 0:
            low_temp += 0.01

        all_temps = self.generate_temperature_list(low_temp, high_temp,
                                                   n_temps, schedule, wavy)

        all_energies = np.empty(nr_eval)
        p_idx = 0
        temp_index = 1
        for cur_temp in all_temps:
            print(f"Iteration {temp_index}/{n_temps} at {cur_temp} degrees", end='\r', flush=True)
            temp_index += 1
            for chain_index in range(chain_length):
                for p in range(self.n_particles):
                    all_energies[p_idx] = self.pot_energy
                    self.do_SA_step(p, cur_temp, forcelist[p_idx])
                    p_idx += 1

        self.write_data(schedule, all_temps, chain_length, all_energies, force, wavy)
        return self.particles

    def total_force_on_particle(self, p):
        F = np.zeros(2)
        for particle in range(self.n_particles):
            if particle != p:
                dir = self.particles[p] - self.particles[particle]
                dist = self.euclidean(self.particles[particle], self.particles[p])
                F += dir / (dist ** 3)
        return F / (self.n_particles - 1)

    def all_forces_on_particle(self, p):
        forces = []
        for particle in range(self.n_particles):
            if particle != p:
                dir = self.particles[p] - self.particles[particle]
                dist = self.euclidean(self.particles[particle], self.particles[p])
                forces.append(dir / (dist ** 3))
        return forces