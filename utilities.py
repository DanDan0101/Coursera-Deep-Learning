import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST


def load_data():
    # Load data
    mndata = MNIST('data_files')
    train_x, train_y = mndata.load_training()
    test_x, test_y = mndata.load_testing()

    # Convert to numpy arrays
    train_x = np.array(train_x).T / 255
    train_y = np.array(train_y).reshape((1, -1))
    test_x = np.array(test_x).T / 255
    test_y = np.array(test_y).reshape((1, -1))

    return train_x, train_y, test_x, test_y


def display(a, index):
    plt.imshow(a.T[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    plt.show()
