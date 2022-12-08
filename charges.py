## Imports
import numpy as np
import numba as nb

from itertools import combinations

# Class definition

class Charges():
    """ Class that handles charge particles and the simulated annealing algorithm
    """

    def __init__(self, n_particles, radius):
        self.n_particles = n_particles
        self.radius = radius
        self.particles = self.generate_points(n_particles, radius)
        self.pot_energy = self.evaluate_configuration()
    
    def generate_points(self, n_particles, radius):
        """ Generate n_particles within a circle 
        """
        points = []
        for n in range(n_particles):
            # random angle and distance from center
            alpha = 2 * np.pi * np.random.random()
            r = radius * np.sqrt(np.random.random())
            # coordinates
            x = r * np.cos(alpha)
            y = r * np.sin(alpha)
            points.append([x,y])

        return np.array(points)


    def evaluate_configuration(self):
        """ Calculate the total energy of the current configuration
        """
        total = 0
        for i,j in list(combinations(range(self.n_particles), 2)):
            p1, p2 = self.particles[i], self.particles[j]
            total += 1 / self.euclidean(p1,p2)

        return total
    

    # @nb.vectorize
    def euclidean(self, p1, p2):
        """ Compute the euclidean distance between two points
        """
        return np.linalg.norm(p2 - p1)


    def check_in_circle(self, p):
        """ Check if point p is within the circle
            returns 0 if it is outside, 1 if it is inside
        """
        distance = self.euclidean([0,0], p)
        if distance > self.radius:
            ## Maybe let it bounce
            return 0
        else:
            return 1


    def move_particle_random(self, p, stepsize=0.01):
        """ Tries to move particle p (index for particles) by stepsize
            returns 1 if succes, 0 if failed
        """
        step = self.generate_points(1, stepsize)[0]
        particle = self.particles[p]
        if self.check_in_circle(particle + step):
            self.particles[p] = particle + step
            return 1
        else:
            return 0


    def do_SA_step(self):
        pass


    def generate_temperature_list(self):
        pass


    def write_data(self):
        pass


    def iterate_SA_optimize(self):
        pass


    def total_force_on_particle(self):
        pass
