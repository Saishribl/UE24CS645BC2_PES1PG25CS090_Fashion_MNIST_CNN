# utils.py

from tensorflow.keras.datasets import fashion_mnist
import numpy as np


def load_data():

    # Load Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Normalize images
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return x_train, y_train, x_test, y_test