import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import functools
from scipy.stats import multivariate_normal

MIN_X = -10
MAX_X = 10
MIN_Y = MIN_X
MAX_Y = MAX_X
LIMIT_ITER = 100


def generate_data(n: int, c: int):
    """
    Generate two matrices of random samples. The dimensions of the matrices are (nc, d+1)
    :param n: The number of samples
    :param m: The number of generated cluster
    :return: data: data with empty label
             data_real: data with correct label
    """
    means = 12*(np.random.rand(c, 2) - 0.5)
    means = means.repeat(n, axis=0)

    data = means + np.random.randn(n*c, 2)

    label_default = np.zeros((n*c, 1))
    label_real = np.repeat(np.array(range(c)), repeats=n).reshape(-1, 1)

    return data, label_default, label_real


def generate_means(k):
    """
    Generate a matrix of labeled random means. The dimension of the matrix is (k, d+1)
    :param k: The number of means
    :return: A matrix of labeled random means
    """
    means = 12*(np.random.rand(k, 2) - 0.5)
    label = np.array([np.array(range(k))]).T
    return means, label


def show_plot(data, label, mu=None, mu_label=None, i=0, save=False):

    plt.scatter(x=data[:, 0], y=data[:, 1], c=label, s=5)
    if mu is not None:
        plt.scatter(x=mu[:, 0], y=mu[:, 1], c=mu_label,
                    marker="*", edgecolors="black", s=100)
    plt.xlim([MIN_X, MAX_X])
    plt.ylim([MIN_Y, MAX_Y])

    if save:
        if not os.path.exists('./output'):
            os.mkdir('./output')
        plt.savefig(f'./output/{i}')
        plt.close()
    else:
        mu_g = np.array([5, 5])
        sigma_g = np.array([[5, 1], [1, 1]])
        X, Y, Z = generate_gaussian(mu_g, sigma_g)
        plt.contour(X, Y, Z)
        plt.show()


def duplicate_mu(N, mu):
    n_mu = mu.shape[0]

    # Duplicate the values of mu along the vertical axis
    mu_v = np.apply_along_axis(functools.partial(
        np.repeat, repeats=N, axis=0), axis=0, arr=mu)

    # Split the array into multiple sub-arrays vertically
    return np.vsplit(mu_v, n_mu)


def assignment(data, mu, mu_label):

    N = data.shape[0]
    mu_duplicated = duplicate_mu(N, mu)

    result = np.zeros((1, N))

    for i in range(mu.shape[0]):
        diff = data-mu_duplicated[i]
        result = np.vstack([result, np.apply_along_axis(
            lambda a: a[0]**2+a[1]**2, 1, diff)])
    result = result[1:]
    arg_min = np.argmin(result, 0)

    mu_label = arg_min
    cost = np.sum(np.min(result, 0))

    return mu_label, cost

# TODO from here we have to change


def get_linked_data(data_label, k):

    sum_data_linked = np.zeros((1, k))
    for label in data_label:
        sum_data_linked[0, int(label)] += 1

    return sum_data_linked


def helper_sum(data, k, data_label):
    result = np.zeros((k, 2))
    result[int(data_label)] = [data[0], data[1]]
    return result


def centroid_update(data, k):
    linked_data = get_linked_data(data_label, k, data_label)

    new_mu = np.apply_along_axis(functools.partial(
        helper_sum, k=k, data_label=data_label), 1, data)

    new_mu = np.sum(new_mu, axis=0)
    new_mu = np.divide(new_mu, linked_data.repeat(2, 0).T, out=np.zeros_like(
        new_mu), where=linked_data.repeat(2, 0).T != 0)
    label = np.array([np.array(range(k))]).T
    new_mu = np.append(new_mu, label, axis=1)

    return new_mu


def cluster(data, data_label, mu, mu_label):

    total_cost = np.array([])

    for i in range(16):

        data, cost = assignment(data, mu, mu_label)
        total_cost = np.append(total_cost, cost)

        new_mu = centroid_update(data, k)
        if np.array_equal(new_mu, mu):
            break
        mu = new_mu
        i += 1
        show_plot(data, mu, i, save=True)

    return data, mu, total_cost


def generate_gaussian(mu, sigma):

    N = 100

    X = np.linspace(MIN_X, MAX_X, N)
    Y = np.linspace(MIN_Y, MAX_Y, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mu, sigma)
    Z = rv.pdf(pos)

    return X, Y, Z


def expectation_step(gaussian: multivariate_normal):
    w = gaussian.pdf()


if __name__ == "__main__":

    k = 2  # The number of clusters we'd like to find
    n = 100  # The size of the data
    c = k  # The number of generated cluster (we should define c=k)
    N = n*c

    if os.path.exists('./output'):
        shutil.rmtree('./output')

    data, data_label, label_real = generate_data(n, c)
    show_plot(data=data, label=label_real, i=-1, save=False)
    mu, mu_label = generate_means(k)
    show_plot(data=data, label=data_label, mu=mu,
              mu_label=mu_label, i=0, save=True)

    data, mu, total_cost = cluster(data, data_label, mu, mu_label)
    print("eof")
