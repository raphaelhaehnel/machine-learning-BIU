import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import functools
import scipy.stats as sc

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
    data_real = data

    label_default = np.zeros((n*c, 1))
    data = np.append(data, label_default, axis=1)

    label_real = np.repeat(np.array(range(c)), repeats=n).reshape(-1, 1)
    data_real = np.append(data_real, label_real, axis=1)

    return data, data_real


def generate_means(k):
    """
    Generate a matrix of labeled random means. The dimension of the matrix is (k, d+1)
    :param k: The number of means
    :return: A matrix of labeled random means
    """
    means = 12*(np.random.rand(k, 2) - 0.5)
    label = np.array([np.array(range(k))]).T
    return np.append(means, label, axis=1)


def generate_sigmas(k):
    """
    Generate k identical matrices of sigma
    :param k: The number of gaussians
    :return: A list of matrices
    """
    return np.repeat(np.eye(2)[None, ...], 2, axis=0)


def generate_alphas(k):
    return np.ones(k)


def show_plot(data, mu=None, i=0, save=False):

    plt.scatter(x=data[:, 0], y=data[:, 1], c=data[:, 2], s=5)
    if mu is not None:
        plt.scatter(x=mu[:, 0], y=mu[:, 1], c=mu[:, 2],
                    marker="*", edgecolors="black", s=100)
    plt.xlim([MIN_X, MAX_X])
    plt.ylim([MIN_Y, MAX_Y])

    if save:
        if not os.path.exists('./output'):
            os.mkdir('./output')
        plt.savefig(f'./output/{i}')
        plt.close()
    else:
        plt.show()


def duplicate_mu(data, mu):
    n_data = data.shape[0]
    n_mu = mu.shape[0]

    # Duplicate the values of mu along the vertical axis
    mu_v = np.apply_along_axis(functools.partial(
        np.repeat, repeats=n_data, axis=0), axis=0, arr=mu[:, :2])

    # Split the array into multiple sub-arrays vertically
    return np.vsplit(mu_v, n_mu)


def assignment(data, mu):

    mu_duplicated = duplicate_mu(data, mu)
    x = data[:, :2]
    result = np.zeros((1, data.shape[0]))

    for i in range(mu.shape[0]):
        diff = x-mu_duplicated[i]
        result = np.vstack([result, np.apply_along_axis(
            lambda a: a[0]**2+a[1]**2, 1, diff)])
    result = result[1:]
    arg_min = np.argmin(result, 0)

    data[:, 2] = arg_min
    cost = np.sum(np.min(result, 0))

    return data, cost


def helper_sum(data, k):
    result = np.zeros((k, 2))
    result[int(data[2])] = [data[0], data[1]]
    return result


def get_linked_data(data, k):

    sum_data_linked = np.zeros((1, k))
    for row in data:
        sum_data_linked[0, int(row[2])] += 1

    return sum_data_linked


def centroid_update(data, k):
    linked_data = get_linked_data(data, k)
    new_mu = np.apply_along_axis(functools.partial(helper_sum, k=k), 1, data)

    new_mu = np.sum(new_mu, axis=0)
    new_mu = np.divide(new_mu, linked_data.repeat(2, 0).T, out=np.zeros_like(
        new_mu), where=linked_data.repeat(2, 0).T != 0)
    label = np.array([np.array(range(k))]).T
    new_mu = np.append(new_mu, label, axis=1)

    return new_mu


def cluster(data, mu):

    total_cost = np.array([])

    for i in range(16):

        data, cost = assignment(data, mu)
        total_cost = np.append(total_cost, cost)

        new_mu = centroid_update(data, k)
        if np.array_equal(new_mu, mu):
            break
        mu = new_mu
        i += 1
        show_plot(data, mu, i, save=True)

    return data, mu, total_cost


def draw_gaussian(X, Y, gaussian_list: list[sc._multivariate.multivariate_normal_frozen]):

    X, Y = np.meshgrid(X, Y)
    pos = np.dstack((X, Y))
    Z_tot = np.zeros(X.shape)

    for gaussian in gaussian_list:
        Z_tot += gaussian.pdf(pos)
    return Z_tot


def generate_list_gaussians(means):

    gaussian_list = []
    sigma_g = np.array([[1, 0], [0, 1]])

    for mu_g in means:
        gaussian_list.append(sc.multivariate_normal(mu_g, sigma_g))

    return gaussian_list


def expectation_step(gaussian: list[sc._multivariate.multivariate_normal_frozen], data, k, alpha):
    x = data[:, :2]

    # Sum for the denominator
    divisor = 0
    for l in range(k):
        divisor += gaussian[l].pdf(x)*alpha[l]

    w_list = []
    for j in range(k):
        w = gaussian[j].pdf(x)*alpha[j] / divisor
        w_list.append(w)

    return w_list


def minimalization_step(w_list):

    for j in range(k):
        alpha = np.
        mu =
        sigma =
        pass


if __name__ == "__main__":

    k = 5  # The number of clusters we'd like to find
    n = 100  # The size of the data
    c = k  # The number of generated cluster (we should define c=k)

    if os.path.exists('./output'):
        shutil.rmtree('./output')

    data, data_real = generate_data(n, c)
    show_plot(data_real, i=-1, save=False)
    mu = generate_means(k)
    sigma = generate_sigmas(k)
    alphas = generate_alphas(k)

    N = 100
    X = np.linspace(MIN_X, MAX_X, N)
    Y = np.linspace(MIN_Y, MAX_Y, N)
    gaussian_list = generate_list_gaussians(mu[:, :2])
    Z = draw_gaussian(X, Y, gaussian_list)
    plt.contour(X, Y, Z)
    plt.show()

    w_list = expectation_step(gaussian_list, data, k, alphas)

    show_plot(data, mu, 0, save=True)

    data, mu, total_cost = cluster(data, mu)
    print("eof")
