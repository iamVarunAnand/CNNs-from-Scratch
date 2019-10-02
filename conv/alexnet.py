# import the necessary packages
from models import Model
from layers.convolutional import Conv2D
from layers.convolutional import MaxPooling2D
from layers.core import Input
from layers.core import Flatten
from layers.core import Dense
from layers.activation import ReLU, Softmax

class AlexNet:
    @staticmethod
    def build(height, width, depth, classes, batch_size):
        # initialize the input shape
        input_shape = (batch_size, height, width, depth)

        # initialize the model
        model = Model()

        # input layer
        model.add(Input(input_shape, name = "input"))

        # first CONV => RELU => POOL block
        model.add(Conv2D(96, (11, 11), padding_type = "same", name = "conv_1-1"))
        model.add(ReLU(name = "relu_1-1"))
        model.add(MaxPooling2D((3, 3), stride = (2, 2), name = "pool_1"))

        # second CONV => RELU => POOL block
        model.add(Conv2D(256, (5, 5), padding_type = "same", name = "conv_2-1"))
        model.add(ReLU(name = "relu_2-1"))
        model.add(MaxPooling2D((3, 3), stride = (2, 2), name = "pool_2"))

        # first (and only) ( CONV => RELU ) * 3 => POOL block
        model.add(Conv2D(384, (3, 3), padding_type = "same", name = "conv-3_1"))
        model.add(ReLU(name = "relu_3-1"))
        model.add(Conv2D(384, (3, 3), padding_type = "same", name = "conv_3-2"))
        model.add(ReLU(name = "relu_3-2"))
        model.add(Conv2D(256, (3, 3), padding_type = "same", name = "conv_3-3"))
        model.add(ReLU(name = "relu_3-3"))
        model.add(MaxPooling2D((3, 3), stride = (2, 2), name = "pool_3"))

        # flatten layer
        model.add(Flatten(name = "flatten"))

        # softmax classifier
        model.add(Dense(classes, name = "dense_1"))
        model.add(Softmax(name = "softmax"))

        # return the constructed network architecture
        return model