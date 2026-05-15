# layers.py

import numpy as np


class ConvLayer:

    def __init__(self, num_filters):

        self.num_filters = num_filters

        # Small random filters
        self.filters = np.random.randn(num_filters, 3, 3) * 0.01

    # Generate all possible 3x3 regions
    def iterate_regions(self, image):

        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):

                image_region = image[i:(i + 3), j:(j + 3)]

                yield image_region, i, j

    # Forward pass
    def forward(self, input):

        self.last_input = input

        h, w = input.shape

        output = np.zeros((h - 2, w - 2, self.num_filters))

        for image_region, i, j in self.iterate_regions(input):

            # Convolution + ReLU
            output[i, j] = np.maximum(
                0,
                np.sum(image_region * self.filters, axis=(1, 2))
            )

        return output

    # Backward pass
    def backward(self, d_L_d_out, learning_rate):

        d_L_d_filters = np.zeros(self.filters.shape)

        for image_region, i, j in self.iterate_regions(self.last_input):

            for f in range(self.num_filters):

                # ReLU derivative
                if d_L_d_out[i, j, f] <= 0:
                    continue

                gradient = d_L_d_out[i, j, f]

                # Skip invalid gradients
                if np.isnan(gradient) or np.isinf(gradient):
                    continue

                d_L_d_filters[f] += gradient * image_region

        # Gradient clipping
        d_L_d_filters = np.clip(d_L_d_filters, -1, 1)

        # Update filters
        self.filters -= learning_rate * d_L_d_filters


class MaxPool:

    # Generate non-overlapping 2x2 regions
    def iterate_regions(self, image):

        h, w, num_filters = image.shape

        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):

                image_region = image[
                    (i * 2):(i * 2 + 2),
                    (j * 2):(j * 2 + 2)
                ]

                yield image_region, i, j

    # Forward pass
    def forward(self, input):

        self.last_input = input

        h, w, num_filters = input.shape

        output = np.zeros((h // 2, w // 2, num_filters))

        for image_region, i, j in self.iterate_regions(input):

            output[i, j] = np.amax(image_region, axis=(0, 1))

        return output

    # Backward pass
    def backward(self, d_L_d_out):

        d_L_d_input = np.zeros(self.last_input.shape)

        for image_region, i, j in self.iterate_regions(self.last_input):

            h, w, f = image_region.shape

            max_values = np.amax(image_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):

                        if image_region[i2, j2, f2] == max_values[f2]:

                            d_L_d_input[
                                i * 2 + i2,
                                j * 2 + j2,
                                f2
                            ] = d_L_d_out[i, j, f2]

        # Remove NaN/Inf
        d_L_d_input = np.nan_to_num(d_L_d_input)

        return d_L_d_input


class FullyConnected:

    def __init__(self, input_len, output_len):

        # Small random weights
        self.weights = np.random.randn(input_len, output_len) * 0.01

        self.biases = np.zeros(output_len)

    # Softmax activation
    def softmax(self, x):

        exp = np.exp(x - np.max(x))

        return exp / np.sum(exp)

    # Forward pass
    def forward(self, input):

        # Save input shape
        self.last_input_shape = input.shape

        # Flatten input
        input = input.flatten()

        self.last_input = input

        # Fully connected operation
        totals = np.dot(input, self.weights) + self.biases

        self.last_totals = totals

        # Apply softmax
        return self.softmax(totals)

    # Stable backward pass
    def backward(self, d_L_d_out, learning_rate):

        # Remove NaN/Inf
        d_L_d_out = np.nan_to_num(d_L_d_out)

        # Gradient for weights
        d_L_d_w = self.last_input[np.newaxis].T @ d_L_d_out[np.newaxis]

        # Gradient for biases
        d_L_d_b = d_L_d_out

        # Gradient for inputs
        d_L_d_inputs = self.weights @ d_L_d_out

        # Gradient clipping
        d_L_d_w = np.clip(d_L_d_w, -1, 1)

        d_L_d_b = np.clip(d_L_d_b, -1, 1)

        d_L_d_inputs = np.clip(d_L_d_inputs, -1, 1)

        # Update weights
        self.weights -= learning_rate * d_L_d_w

        self.biases -= learning_rate * d_L_d_b

        return d_L_d_inputs.reshape(self.last_input_shape)