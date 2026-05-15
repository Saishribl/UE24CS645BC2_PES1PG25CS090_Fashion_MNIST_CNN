# cnn.py

from layers import ConvLayer, MaxPool, FullyConnected


class CNN:

    def __init__(self):

        self.conv = ConvLayer(32)

        self.pool = MaxPool()

        self.fc = FullyConnected(13 * 13 * 32, 10)