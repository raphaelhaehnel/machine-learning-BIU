import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import functools
from scipy.stats import multivariate_normal

w = np.array([[1],
              [2],
              [3]])
w1 = np.array([1, 2, 3])

x = np.array([[10, 20],
              [30, 40],
              [50, 60]])

mu = np.array([1, 2])

print(w*x)
print("w: ", w.shape)
print("w1: ", w1.shape)
print(x.shape)
