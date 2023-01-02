import scipy.io
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt


def helper_retrieve(data):
    """
    """
    a1 = np.split(data, 15)
    a2 = np.array(a1)
    a31 = a2[:, 0:8, :]
    a32 = a2[:, 8:11, :]

    train = np.concatenate(a31)
    test = np.concatenate(a32)

    return train, test


def retrieve_data():
    mat = scipy.io.loadmat('facesData.mat')
    faces = mat.get('faces')
    labels = mat.get('labeles')

    X_train, X_test = helper_retrieve(faces)
    y_train, y_test = helper_retrieve(labels)

    return X_train, X_test, y_train, y_test


def extract_eigenvectors(X: np.ndarray):
    A = np.matmul(X.T, X)
    w, v = np.linalg.eig(A)
    return w, v


if __name__ == "__main__":

    # Creating training and testing sets
    X_train, X_test, y_train, y_test = retrieve_data()

    # Find the mean image
    mean = X_train.mean(0)

    # Find the eigenvectors
    w, v = extract_eigenvectors(X_train)

    plt.imshow(mean.reshape((32, 32)).T, cmap='gray')
    plt.show()

    for i in range(10):
        plt.imshow(X_train[i].reshape((32, 32)).T, cmap='gray')
        plt.show()
