# import the necessary packages
from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    # initialize a class variable to keep track of the number of layers initialized
    layer_index = 0

    def __init__(self, name = None):
        # call the parent class constructor
        super(Layer, self).__init__()

        # update the layer index
        Layer.layer_index = Layer.layer_index + 1

        # set the layer name
        if name is None:
            self.name = "layer_" + str(Layer.layer_index)
        else:
            self.name = name

        # initialize other variables
        self.input_shape = None
        self.output_shape = None
        self.weights = None
        self.bias = None
        self.cache = None
        self.gradients = {"W": None, "b" : None}

    def set_input_shape(self, input_shape = None):
        # set the layer's input shape
        self.input_shape = input_shape

    def set_gradients(self):
        # set the corresponding dictionary keys with zero initialized arrays
        try:
            self.gradients["W"] = np.zeros(self.weights.shape)
            self.gradients["b"] = np.zeros(self.bias.shape)
        except:
            self.gradients["W"] = None
            self.gradients["b"] = None

    def update_weights(self, lr):
        # update the layer weights
        try:
            self.weights -= lr * self.gradients["W"]
            self.bias -= lr * self.gradients["b"]
        except:
            pass

    @abstractmethod
    def set_output_shape(self):
        pass

    @abstractmethod
    def init_layer(self, input_shape = None):
        pass

    @abstractmethod
    def forward_call(self, input):
        pass

    @abstractmethod
    def backward_call(self, input):
        pass
    #
    # @abstractmethod
    # def get_weights_shape(self):
    #     pass
    #
    # @abstractmethod
    # def get_bias_shape(self):
    #     pass
