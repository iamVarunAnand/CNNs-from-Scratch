# import the necessary packages
from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    # initialize a class variable to keep track of the number of layers initialized
    layer_index = 0

    def __init__(self, name = None):
        # call the parent class constructor
        super().__init__()

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

    @abstractmethod
    def set_input_shape(self, input_shape = None):
        pass

    @abstractmethod
    def set_output_shape(self):
        pass

    @abstractmethod
    def init_layer(self, input_shape = None):
        pass

    # @abstractmethod
    # def call_forward(self):
    #     pass
    #
    # @abstractmethod
    # def call_backward(self):
    #     pass
    #
    # @abstractmethod
    # def get_weights_shape(self):
    #     pass
    #
    # @abstractmethod
    # def get_bias_shape(self):
    #     pass
