import matplotlib.pyplot as plt
import numpy as np
from plotting import Window

population = [1, 2, 3, 4, 5]
weights_list = [
    [5, 1, 3, 2, 1],
    [1, 1, 2, 5, 5],
    [1, 1, 5, 1, 1],
    [1, 3, 1, 3, 1],
    [1, 2, 3, 2, 1],
]
w = Window(population, weights_list)

plt.show()