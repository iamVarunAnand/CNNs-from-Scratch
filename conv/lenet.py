# import the necessary packages
from models import Model
from layers.core import Input
from layers.convolutional import Conv2D, MaxPooling2D
from layers.activation import ReLU, Softmax
from layers.core import Flatten, Dense

class LeNet:
    @staticmethod
    def build(height, width, depth, classes, batch_size):
        # construct the input shape tuple
        input_shape = (batch_size, height, width, depth)

        # instantiate the model
        model = Model()

        # input layer
        model.add(Input(input_shape = input_shape, name = "input"))

        # first CONV => RELU => POOL block
        model.add(Conv2D(20, (5, 5), padding_type = "same", name = "conv_1"))
        model.add(ReLU(name = "relu_1"))
        model.add(MaxPooling2D(kernel_size = (2, 2), stride = (2, 2), name = "pool_1"))

        # second CONV => RELU => POOL block
        model.add(Conv2D(50, (5, 5), padding_type = "same", name = "conv_2"))
        model.add(ReLU(name = "relu_2"))
        model.add(MaxPooling2D(kernel_size = (2, 2), stride = (2, 2), name = "pool_2"))

        # first and only set of FC => RELU layers
        model.add(Flatten(name = "flatten"))
        model.add(Dense(500, name = "fc_1"))
        model.add(ReLU(name = "relu_3"))

        # softmax classifier
        model.add(Dense(classes, name = "fc_2"))
        model.add(Softmax(name = "softmax"))

        # return the constructed network architecture
        return model