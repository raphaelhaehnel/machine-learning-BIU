import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import functools
from scipy.stats import multivariate_normal

# x = np.random.rand(100, 2)
# plt.scatter(x=x[:, 0], y=x[:, 1], s=5)
# plt.xlim([-10, 10])
# plt.ylim([-10, 10])
# plt.show()

# sigma = np.array([[1, -1], [-1, 1]])
# x = np.matmul(x, sigma)
# plt.scatter(x=x[:, 0], y=x[:, 1], s=5)
# plt.xlim([-10, 10])
# plt.ylim([-10, 10])
# plt.show()


def generate_specific_means():
    means = np.array([[-3, 1], [-3, -1]])
    sigma = np.array([[6, 0], [0, 0.8]])
    x = np.random.rand(100, 2)

    x = np.matmul(x, sigma)
    x1 = x + means[0]
    x2 = x + means[1]

    return np.vstack([x1, x2])


x = generate_specific_means()
plt.scatter(x=x[:, 0], y=x[:, 1], s=5)
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.show()
