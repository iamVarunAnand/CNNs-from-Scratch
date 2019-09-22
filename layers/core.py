# import the necessary packages
from .layer import Layer
import numpy as np


class Input(Layer):
    """
    The input layer is merely a wrapper to organize the data flow for the rest of the network. It
    simply passes the input, unchanged, to the subsequent layer.
    """

    def __init__(self, input_shape, name = None):
        # call the parent class constructor
        super(Input, self).__init__(name)

        # set the input shape for the input layer
        self.input_shape = input_shape

    def set_output_shape(self):
        # set the output shape of the input layer equal to the input shape
        self.output_shape = self.input_shape

    def init_layer(self, input_shape = None):
        print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight and bias matrices
        self.weights = None
        self.bias = None

        # set the output shapes
        self.set_output_shape()

        # set the layer's gradients
        self.set_gradients()

        print(self.name, "output_shape = ", self.output_shape, sep = " ")

    def forward_call(self, input):
        # return the input unchanged
        return input

    def backward_call(self, input):
        # return the input unchanged (provided only for completeness)
        return input

class Dense(Layer):
    """
    The Dense layer represents the fully connected block. Input tensor is multiplied by the
    layer's weight matrix and the output tensor is returned.
    """
    def __init__(self, units, name = None):
        # call the parent class constructor
        super(Dense, self).__init__(name)
        self.units = units

    def set_output_shape(self):
        # set the layer's output shape
        self.output_shape = (self.units, 1)

    def init_layer(self, input_shape = None):
        # set the input shapes
        self.set_input_shape(input_shape = input_shape)

        print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight and bias matrices
        self.weights = np.random.randn(self.units, self.input_shape[0])
        self.bias = np.zeros((self.units, 1))

        # set the output shapes
        self.set_output_shape()

        # set the layer's gradients
        self.set_gradients()

        print(self.name, "output_shape = ", self.output_shape, sep = " ")

    def forward_call(self, input):
        # calculate the output
        output = np.add(np.dot(self.weights, input), self.bias)

        # initialize the layer's cache
        self.cache = (input, output)

        # return the calculated value
        return output

    def backward_call(self, input):
        # extract the values from the cache
        (A_prev, z) = self.cache

        # calculate the gradients
        output = np.dot(self.weights.T, input)
        self.gradients["W"] = np.dot(input, A_prev.T)
        self.gradients["b"] = input

        # return the calculated value
        print("[INFO] backprop output shape at layer {}: {}".format(self.name, output.shape))
        return output


class Flatten(Layer):
    """
    Flattens the input along the axis specified.
    """
    def __init__(self, name = None):
        # call the parent class constructor
        super(Flatten, self).__init__(name)

    def set_output_shape(self):
        # calculate the total number of elements in the input tensor
        total_elts = 1
        for elts in self.input_shape:
            total_elts = total_elts * elts

        # set the layer's output shape
        self.output_shape = (total_elts, 1)

    def init_layer(self, input_shape = None):
        # set the input shapes
        self.set_input_shape(input_shape)

        print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight and bias matrices
        self.weights = None
        self.bias = None

        # set the output shapes
        self.set_output_shape()

        # set the layer's gradients
        self.set_gradients()

        print(self.name, "output_shape = ", self.output_shape, sep = " ")

    def forward_call(self, input):
        # reshape the input tensor
        output = np.reshape(input, self.output_shape)

        # return the calculated value
        return output

    def backward_call(self, input):
        # reshape the input tensor
        output = np.reshape(input, self.input_shape)

        # return the calculated value
        print("[INFO] backprop output shape at layer {}: {}".format(self.name, output.shape))
        return output

