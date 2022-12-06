import os
import matplotlib.pyplot as plt
import numpy as np
import functools
import scipy.io
import shutil

LIMIT_ITER = 100


def generate_data(n, c, dim):
    means = 12*(np.random.rand(c, dim) - 0.5)
    means = means.repeat(n, axis=0)

    data = means + np.random.randn(n*c, dim)
    col = np.zeros((n*c, 1))
    data = np.append(data, col, axis=1)
    return data


def generate_means(k, dim):
    means = 12*(np.random.rand(k, dim) - 0.5)
    label = np.array([np.array(range(k))]).T
    return np.append(means, label, axis=1)


def show_plot(data, mu, i, save, dim):
    if dim == 2:
        plt.scatter(x=data[:, 0], y=data[:, 1], c=data[:, 2], s=20)
        plt.scatter(x=mu[:, 0], y=mu[:, 1], c=mu[:, 2],
                    marker="*", edgecolors="black", s=100)
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 3], s=20)
        ax.scatter(mu[:, 0], mu[:, 1], mu[:, 2], c=mu[:, 3],
                   marker="*", edgecolors="black", s=100)

    if save:
        if not os.path.exists('./output'):
            os.mkdir('./output')
        plt.savefig(f'./output/{i}')
        plt.close()
    else:
        plt.show()


def duplicate_mu(data, mu, dim):
    n_data = data.shape[0]
    n_mu = mu.shape[0]

    # Duplicate the values of mu along the vertical axis
    mu_v = np.apply_along_axis(functools.partial(
        np.repeat, repeats=n_data, axis=0), axis=0, arr=mu[:, :dim])

    # Split the array into multiple sub-arrays vertically
    return np.vsplit(mu_v, n_mu)


def assignment(data, mu, dim):

    mu_duplicated = duplicate_mu(data, mu, dim)
    x = data[:, :dim]
    result = np.zeros((1, data.shape[0]))

    for i in range(mu.shape[0]):
        diff = x-mu_duplicated[i]
        result = np.vstack([result, np.apply_along_axis(
            lambda a: a[0]**2+a[1]**2, 1, diff)])
    result = result[1:]
    arg_min = np.argmin(result, 0)

    data[:, dim] = arg_min

    return data


def helper_sum(data, k, dim):
    result = np.zeros((k, dim))
    if dim == 2:
        result[int(data[dim])] = [data[0], data[1]]
    if dim == 3:
        result[int(data[dim])] = [data[0], data[1], data[2]]

    return result


def get_linked_data(data, k, dim):

    sum_data_linked = np.zeros((1, k))
    for row in data:
        sum_data_linked[0, int(row[dim])] += 1

    return sum_data_linked


def centroid_update(data, k, dim):
    linked_data = get_linked_data(data, k, dim)
    new_mu = np.apply_along_axis(
        functools.partial(helper_sum, k=k, dim=dim), 1, data)  # dim is not supposed to be an array

    new_mu = np.sum(new_mu, axis=0)
    new_mu = np.divide(new_mu, linked_data.repeat(dim, 0).T, out=np.zeros_like(
        new_mu), where=linked_data.repeat(dim, 0).T != 0)
    label = np.array([np.array(range(k))]).T
    new_mu = np.append(new_mu, label, axis=1)

    return new_mu


def cluster(data, mu):
    pass


def retrive_data():
    mat = scipy.io.loadmat('mnist_all.mat')

    x1 = mat.get('train1')
    x2 = mat.get('train2')
    X_train = np.concatenate((x1, x2), axis=0)

    y1 = np.zeros(len(x1), dtype=int)
    y2 = np.ones(len(x2), dtype=int)
    y_train = np.concatenate((y1, y2), axis=0)

    x1_test = mat.get('test1')
    x2_test = mat.get('test2')
    X_test = np.concatenate((x1_test, x2_test), axis=0)

    y1_test = np.zeros(len(x1_test), dtype=int)
    y2_test = np.ones(len(x1_test), dtype=int)
    y_test = np.concatenate((y1_test, y2_test), axis=0)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":

    k = 10
    n = 100
    c = k
    dim = 2

    data = generate_data(n, c, dim)
    mu = generate_means(k, dim)

    X_train, y_train, X_test, y_test = retrive_data()

    if os.path.exists('./output'):
        shutil.rmtree('./output')

    for i in range(LIMIT_ITER):

        data = assignment(data, mu, dim)
        new_mu = centroid_update(data, k, dim)
        if np.array_equal(new_mu, mu):
            break
        mu = new_mu
        show_plot(data, mu, i, save=True, dim=dim)
