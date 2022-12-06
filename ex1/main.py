###########################################################
# FILE : main.py
# WRITER : Raphael Haehnel
# DESCRIPTION: Introduction to Machine Learning (BIU), ex1
###########################################################

import numpy as np
import matplotlib as plt
import plotly.express as px


def generate_x(n: int, m: int):
    """
    Generate a random matrix of dimension n x m
    :param n: The number of samples
    :param m: The number of features
    :return: The matrix X
    """
    return np.random.uniform(size=(n, m))


def generate_beta(m: int):
    """
    Generate the weight matrix
    :param m: The number of features
    :return: The matrix beta
    """
    return np.array([np.arange(1, m+1, 1)]).T


def generate_epsilon(n: int, sigma: float = 1.0):
    """
    Generate a random vector
    :param n: Size of the vector
    :param sigma: The standard deviation of the random generated elements
    :return: A vector of random elements
    """
    return np.random.normal(0, sigma, size=(n, 1))


def compute_y(x: np.ndarray, beta: np.ndarray, epsilon: np.ndarray):
    """
    Compute the y values of the training example
    :param x: The x values of the training example
    :param beta: The vector of parameters
    :param epsilon: The vector of random values
    :return: A vector of values
    """
    return np.dot(x, beta) + epsilon


def solve_beta(x: np.ndarray, y: np.ndarray):
    """
    Find the beta parameters of the linear regression
    :param x: Training example
    :param y: Training example
    :return: The parameter vector beta of the regression model
    """
    a1 = np.linalg.inv(np.dot(x.T, x))
    a2 = np.dot(x.T, y)
    return np.dot(a1, a2)


if __name__ == '__main__':
    n = 100
    m = 2
    sigma = 0.01

    X = generate_x(n, m)
    beta = generate_beta(m)
    epsilon = generate_epsilon(n, sigma)
    Y = compute_y(X, beta, epsilon)
    new_beta = solve_beta(X, Y)

    # If there is only one feature, we can print the data.
    if m == 1:
        fig = px.scatter(x=X.flatten(), y=Y.flatten(), title=f"Y(X) for m={m}, \u03C3={sigma}")
        fig.show()

    print("beta")
    print(beta)
    print("new beta")
    print(new_beta)
