import scipy.io
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mat = scipy.io.loadmat('mnist_all.mat')
    data = mat.get('train0')
    for element in data:
        array = np.array(element)
        image = np.reshape(array, (28, 28))
        plt.imshow(image)
        plt.show()

    print("EOP")
