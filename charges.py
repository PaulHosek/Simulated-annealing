## Imports
import numpy as np
import numba as nb

# Class definition

class Charges():
    """ Class that handles charge particles and the simulated annealing algorithm
    """

    def __init__(self, n_particles, radius):
        self.particles = self.initialise_points(n_particles, radius)

    
    def initialise_points(self, n_particles, radius):
        """ Generate n_particles within a circle 
        """
        points = []
        for n in range(n_particles):
            # random angle
            alpha = 2 * np.pi * np.random.random()
            r = radius * np.sqrt(np.random.random())
            # calculating coordinates
            x = r * np.cos(alpha)
            y = r * np.sin(alpha)
            points.append([x,y])

        return np.array(points)


    def evaluate_constellation():
        pass

    
    @nb.vectorize
    def calculate_euclidean_distance():
        pass


    def move_particle_random():
        pass


    def do_SA_step():
        pass


    def generate_temperature_list():
        pass


    def write_data():
        pass


    def iterate_SA_optimise():
        pass


    def total_force_on_particle():
        pass
