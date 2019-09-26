# import the necessary packages
from .layer import *
import numpy as np

class ReLU(Layer):
    """
    Applies the ReLU activation function on the input tensor
    """
    def __init__(self, name = None):
        # call the parent constructor
        super(ReLU, self).__init__(name)

    def set_output_shape(self):
        # set the layer's output shape
        self.output_shape = self.input_shape

    def init_layer(self, input_shape = None):
        # set the layer's input shape
        self.set_input_shape(input_shape)

        # print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight and bias matrices
        self.weights = None
        self.bias = None

        # set the layer's output shape
        self.set_output_shape()

        # print(self.name, "output_shape = ", self.output_shape, sep = " ")

        # set the layer's gradients
        self.set_gradients()

    def forward_call(self, input):
        # apply the activation function
        output = np.maximum(input, 0)

        # initialize the cache
        self.cache = output

        # return the calculated value
        return output

    def backward_call(self, input):
        # extract the tensors from the cache
        A_prev = self.cache

        # calculate the gradients
        A_prev[A_prev != 0] = 1
        output = A_prev * input

        # return the calculated value
        # print("[INFO] backprop output shape at layer {}: {}".format(self.name, output.shape))
        # print("[INFO] Mean of backprop gradient at layer {}: {}".format(self.name, np.mean(output)))
        return output

class Softmax(Layer):
    def __init__(self, name = None):
        # call the parent class constructor
        super(Softmax, self).__init__(name)

    def set_output_shape(self):
        # set the layer's output shape
        self.output_shape = self.input_shape

    def init_layer(self, input_shape = None):
        # set the layer's input shape
        self.set_input_shape(input_shape)

        # print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # set the layer's weight and bias matrices
        self.weights = None
        self.bias = None

        # set the layer's output shape
        self.set_output_shape()

        # print(self.name, "output_shape = ", self.output_shape, sep = " ")

        # initialize the layer's gradients
        self.set_gradients()

    def forward_call(self, input):
        # stabilize the input
        max = np.max(input, axis = 1)
        input = input - np.expand_dims(max, axis = -1)

        # apply the activation function on the input tensor
        exp_input = np.exp(input)
        output = exp_input / np.expand_dims(np.sum(exp_input, axis = 1), axis = -1)

        # return the calculated value
        return output

    def backward_call(self, input):
        # function ignored as derivative is directly calculated for the last output layer
        pass