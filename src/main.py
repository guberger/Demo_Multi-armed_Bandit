import matplotlib.pyplot as plt
import numpy as np
from plotting import Window

population = [1, 2, 3, 4, 5]
weights_list = [
    [1, 2, 3, 2, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
]
w = Window(population, weights_list, 1)

plt.show()