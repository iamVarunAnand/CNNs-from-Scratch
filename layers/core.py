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
        super().__init__(name)

        # set the input shape for the input layer
        self.input_shape = input_shape

    def set_input_shape(self, input_shape = None):
        # method provided only for completeness, input shape is set by class constructor
        pass

    def set_output_shape(self):
        # set the output shape of the input layer equal to the input shape
        self.output_shape = self.input_shape

    def init_layer(self, input_shape = None):
        # set the input shapes
        self.set_input_shape(input_shape)

        print(self.__class__.__name__, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight and bias matrices
        self.weights = None
        self.bias = None

        # set the output shapes
        self.set_output_shape()

        print(self.__class__.__name__, "output_shape = ", self.output_shape, sep = " ")

class Dense(Layer):
    def __init__(self, units, name = None):
        # call the parent class constructor
        super().__init__(name)
        self.units = units

    def set_input_shape(self, input_shape = None):
        # set the input shape
        self.input_shape = input_shape

    def set_output_shape(self):
        # set the layer's output shape
        self.output_shape = (self.units, 1)

    def init_layer(self, input_shape = None):
        # set the input shapes
        self.set_input_shape(input_shape = input_shape)
        # print(self.__class__.__name__, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight and bias matrices
        self.weights = np.random.randn(self.units, self.input_shape[0])
        self.bias = np.random.randn(self.units, 1)
        # print(self.__class__.__name__, "weight dimensions = ", self.weights.shape, self.bias.shape,
        #       sep = " ")

        # set the output shapes
        self.set_output_shape()
        # print(self.__class__.__name__, "output_shape = ", self.output_shape, sep = " ")

# class Flatten(Layer):
#     def __init(self):
