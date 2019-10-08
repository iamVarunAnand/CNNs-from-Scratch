# import the necessary packages
from .layer import Layer
import multiprocessing as mp
import numpy as np


class Conv2D(Layer):
    """
    Implements the convolutional layer. Filters are applied to the input tensor which is then
    transformed into the appropriate output tensor.
    """
    def __init__(self, filters, kernel_size, stride = (1, 1), padding_type = "same", name = None):
        # call the parent classZ constructor
        super(Conv2D, self).__init__(name)

        # set the layer attributes
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.offset = (kernel_size[0] // 2, kernel_size[1] // 2)

        # calculate the padding, if any
        if padding_type == "same":
            self.padding = self.offset
        else:
            self.padding = None

    def set_output_shape(self):
        # set the layer's output shape
        self.output_shape = (self.input_shape[0], self.input_shape[1], self.filters)

    def init_layer(self, input_shape = None):
        # set the layer's input_shape
        self.set_input_shape(input_shape)

        # print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight and bias matrices
        self.weights = np.random.randn(self.filters, self.kernel_size[0], self.kernel_size[1],
                                       self.input_shape[-1]) * 0.01
        self.weights = self.weights[np.newaxis, : ,: , :, :].astype(np.float16)
        self.bias = np.zeros((self.filters, 1), dtype = np.float16)

        # set the layer's output shape
        self.set_output_shape()

        # set the layer's gradients
        self.set_gradients()

        # print(self.name, "output_shape = ", self.output_shape, sep = " ")

    def __pad_input(self, input):
        (hp, wp) = self.padding

        return np.pad(input, pad_width = [(0, 0), (hp, hp), (wp, wp), (0, 0)], mode = "constant",
                      constant_values = (0, 0))

    def __convolve(self, arr_1, arr_2):
        return np.sum(np.multiply(arr_1, arr_2), axis = (1, 2, 3))

    def forward_call_parallel(self, input):
        # calculate the output value
        output = np.sum(np.multiply(input, self.weights), axis = (-1, -2, -3))

        # return the calculated value
        return output

    def forward_call(self, input):
        # initialize the multiprocessing pool
        pool = mp.Pool(8)

        # initialize the layer's cache
        self.cache = input

        # expand the dimensions of the input to suit multiplication
        input = input[:, :, np.newaxis,: ,: ,:]

        # execute the forward call
        output = pool.map(self.forward_call_parallel, input)

        # close the multiprocessing object
        pool.close()
        pool.join()

        # return the calculated value
        return np.array(output)

    def backward_call_parallel(self, iter):
        # extract the values
        (cache, inpt) = iter

        # calculate the total number of training examples
        m = inpt.shape[0]

        # calculate the output value
        output = np.sum(np.multiply(inpt, self.weights), axis = 1)

        # calculate the gradient values
        self.gradients["W"] += (1 / m) * np.sum(np.multiply(inpt, cache), axis = 0)
        self.gradients["b"] += (1 / m) * np.sum(np.squeeze(inpt), axis = 0, keepdims = True).T

        return output

    def backward_call(self, input):
        # initialize the multiprocessing pool
        pool = mp.Pool(8)

        # expand the dimensions of the input to suit multiplication
        input = input[:, :, :, np.newaxis, np.newaxis, np.newaxis]

        # expand the dimensions of the cache to suit multiplication
        self.cache = self.cache[:, :, np.newaxis, :, :, :]

        # initialize the output
        output = pool.map(self.backward_call_parallel, zip(self.cache, input))

        # close the multiprocessing object
        pool.close()
        pool.join()

        # return the calculated value
        return np.array(output)

class MaxPooling2D(Layer):
    """
    This layer downsamples the input tensor based on the given pooling window size
    """
    def __init__(self, kernel_size, stride = (2, 2), name = None):
        # call the parent class constructor
        super(MaxPooling2D, self).__init__(name)

        # set the layer attributes
        self.kernel_size = kernel_size
        self.offset = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.stride = stride

    def set_output_shape(self):
        # set the layer's output shape
        self.output_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[-1])

    def init_layer(self, input_shape = None):
        # set the layer's input shape
        self.set_input_shape(input_shape)

        # print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight and bias matrices
        self.weights = None
        self.bias = None

        # set the layer's output shape
        self.set_output_shape()

        # set the layer's gradients
        self.set_gradients()

        # print(self.name, "output_shape = ", self.output_shape, sep = " ")

    def forward_call_parallel(self, input):
        # calculate the output value
        output = np.amax(input, axis = (1, 2))

        # return the calculated value
        return output

    def forward_call(self, input):
        # initialize the multiprocessing pool
        pool = mp.Pool(8)

        # execute the forward call
        output = pool.map(self.forward_call_parallel, input)

        # close the multiprocessing object
        pool.close()
        pool.join()

        # initialize the layer's cache
        self.cache = input

        # return the calculated value
        return np.array(output)

    def __create_mask(self, arr):
        return (arr == np.max(arr))

    def backward_call_parallel(self, iter):
        # extract the values
        (cache, inpt) = iter

        # calculate the gradients
        mask = self.__create_mask(cache)
        output = np.multiply(mask, inpt)

        # return the calculated value
        return output

    def backward_call(self, input):
        # initialize the multiprocessing pool
        pool = mp.Pool(8)

        # expand the dimensions of the input to suit multiplication
        input = input[:, :, np.newaxis, np.newaxis, :]

        # execute the forward call
        output = pool.map(self.backward_call_parallel, zip(self.cache, input))

        # close the multiprocessing object
        pool.close()
        pool.join()

        # return the calculated value
        return np.array(output)
