from charges import Charges
from plotting import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n = 12
    method = 'exponential_0.003'
    iterations = 100
    markov = 10

    ch = Charges(n)
    animate_convergence(ch, 0.1,50, iterations, method, markov, True)
    # plt.show()