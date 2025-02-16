import os
import matplotlib.pyplot as plt
import numpy as np
import functools


def generate_data(n, c):
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

    means = 12*(np.random.rand(k, 2) - 0.5)
    label = np.array([np.array(range(k))]).T
    return np.append(means, label, axis=1)


def show_plot(data, mu=None, i=0, save=False):

    plt.scatter(x=data[:, 0], y=data[:, 1], c=data[:, 2], s=20)
    if mu is not None:
        plt.scatter(x=mu[:, 0], y=mu[:, 1], c=mu[:, 2],
                    marker="*", edgecolors="black", s=100)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])

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


def cost_function(data, mu):
    pass


def cluster(data, mu):

    total_cost = np.array([])

    for i in range(16):

        data, cost = assignment(data, mu)
        total_cost = np.append(total_cost, cost)
        show_plot(data, mu, i, save=True)
        mu = centroid_update(data, k)
        i += 1
        show_plot(data, mu, i, save=True)

    return data, mu, total_cost


if __name__ == "__main__":

    k = 8  # The number of clusters we'd like to find
    n = 100  # The size of the data
    c = k  # The number of generated cluster (we should define c=k)

    data, data_real = generate_data(n, c)
    show_plot(data_real, i=-1, save=False)
    mu = generate_means(k)
    show_plot(data, mu, 0, save=True)

    data, mu, total_cost = cluster(data, mu)
    print("eof")
