# import the necessary packages
from project.models import Model
from project.layers.convolutional import Conv2D, MaxPooling2D
from project.layers.core import Input, Dense, Flatten
from project.layers.activation import ReLU, Softmax

class VGG16:
    @staticmethod
    def build(height, width, depth, classes):
        # calculate the input shape
        input_shape = (height, width, depth)

        # instantiate the model
        model = Model()

        # input layer
        model.add(Input(input_shape = input_shape, name = "input"))

        # first CONV => RELU => CONV => RELU => POOl block
        model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding_type = "same",
                         name = "conv_1-1"))
        model.add(ReLU(name = "relu_1-1"))
        model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding_type = "same",
                         name = "conv_1-2"))
        model.add(ReLU(name = "relu_1-2"))
        model.add(MaxPooling2D(kernel_size = (2, 2), stride = (2, 2), name = "pool_1"))

        # second CONV => RELU => CONV => RELU => POOl block
        model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding_type = "same",
                         name = "conv_2-1"))
        model.add(ReLU(name = "relu_2-1"))
        model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding_type = "same",
                         name = "conv_2-2"))
        model.add(ReLU(name = "relu_2-2"))
        model.add(MaxPooling2D(kernel_size = (2, 2), stride = (2, 2), name = "pool_2"))

        # third CONV => RELU => CONV => RELU => POOl block
        model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding_type = "same",
                         name = "conv_3-1"))
        model.add(ReLU(name = "relu_3-1"))
        model.add(Conv2D(filters = 256, kernel_size = (3, 3), padding_type = "same",
                         name = "conv_3-2"))
        model.add(ReLU(name = "relu_3-2"))
        model.add(MaxPooling2D(kernel_size = (2, 2), stride = (2, 2), name = "pool_3"))

        # first CONV => RELU => CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding_type = "same",
                         name = "conv_4-1"))
        model.add(ReLU(name = "relu_4-1"))
        model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding_type = "same",
                         name = "conv_4-2"))
        model.add(ReLU(name = "relu_4-2"))
        model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding_type = "same",
                         name = "conv_4-2"))
        model.add(ReLU(name = "relu_4-2"))
        model.add(MaxPooling2D(kernel_size = (2, 2), stride = (2, 2), name = "pool_4"))

        # second CONV => RELU => CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding_type = "same",
                         name = "conv_5-1"))
        model.add(ReLU(name = "relu_5-1"))
        model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding_type = "same",
                         name = "conv_5-2"))
        model.add(ReLU(name = "relu_5-2"))
        model.add(Conv2D(filters = 512, kernel_size = (3, 3), padding_type = "same",
                         name = "conv_5-2"))
        model.add(ReLU(name = "relu_5-2"))
        model.add(MaxPooling2D(kernel_size = (2, 2), stride = (2, 2), name = "pool_5"))

        # flatten layer
        model.add(Flatten())

        # softmax classifier
        model.add(Dense(units = 10, name = "dense_1"))
        model.add(Softmax(name = "softmax_1"))

        # return the constructed model architecture
        return model