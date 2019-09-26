# import the necessary packages
from .layer import Layer
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
        # get the height and width from the layer's input shape attribute
        (h, w) = self.input_shape[:2]

        # get the kernel dimensions from the layer's kernel size attribute
        (kh, kw) = self.kernel_size

        # get the padding dimensions from the layer's padding attribute
        (ph, pw) = self.padding

        # get the stride dimension from the layer's stride attribute
        (sh, sw) = self.stride

        # calculate the output dimensions
        oh = ((h + (2 * ph) - kh) // sh) + 1
        ow = ((w + (2 * pw) - kw) // sw) + 1
        od = self.filters

        # set the layer's output shape
        self.output_shape = (oh, ow, od)

    def init_layer(self, input_shape = None):
        # set the layer's input_shape
        self.set_input_shape(input_shape)

        # print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight and bias matrices
        self.weights = np.random.randn(self.filters, self.kernel_size[0], self.kernel_size[1],
                                       self.input_shape[-1]) * 0.01
        self.bias = np.zeros((self.filters, 1))

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

    def forward_call(self, input):
        # calculate the number of training examples
        m = input.shape[0]

        # initialize the output tensor
        output = np.zeros((m, self.output_shape[0], self.output_shape[1], self.output_shape[2]))

        # pad the input based on the padding attribute
        if self.padding is not None:
            input = self.__pad_input(input)

        # expand the dimensions of the weight tensor to suit multiplication
        weights = self.weights[np.newaxis, :, :, :, :]

        # loop over the output
        for i in range(0, output.shape[1]):
            for j in range(0, output.shape[2]):
                # define the sub input limits
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                # extract the sub input
                sub_input = input[:, h_start: h_end, w_start: w_end, :]

                # expand the sub input tensor for multiplication
                sub_input = sub_input[:, np.newaxis, :, :, :]

                # calculate the output value
                output[:, i, j, :] = np.sum(np.multiply(sub_input, weights),
                                                           axis = (-1, -2, -3))

        # initialize the layer's cache
        self.cache = input

        # return the calculated value
        return output

    def backward_call(self, input):
        # calculate the number of training examples
        m = input.shape[0]

        # retrieve the information from the cache
        A_prev = self.cache

        # initialize the output
        output = np.zeros((m, self.input_shape[0], self.input_shape[1], self.input_shape[2]))

        # apply padding to A_prev and the output
        A_prev_pad = self.__pad_input(A_prev)
        output_pad = self.__pad_input(output)

        # expand the dimensions of the weight tensor to suit multiplication
        weights = self.weights[np.newaxis, :, :, :, :]

        # extract the offset values
        (ho, wo) = self.offset

        # loop through the input
        for i in range(0, input.shape[1]):
            for j in range(0, input.shape[2]):
                # define the sub input limits
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                # extract the sub input
                sub_input = input[:, i, j, :]

                # reshape and expand the sub input to suit multiplication
                sub_input = sub_input[:, :, np.newaxis, np.newaxis, np.newaxis]

                # calculate the gradient value
                output_pad[:, h_start: h_end, w_start: w_end, :] += np.sum(
                    np.multiply(sub_input, weights), axis = 1)

                # extract the sub prev
                sub_prev = A_prev_pad[:, h_start : h_end, w_start : w_end, :]

                # reshape and expand the sub prev to suit multiplication
                sub_prev = sub_prev[:, np.newaxis, :, :, :]

                self.gradients["W"] += (1 / m) * np.sum(np.multiply(sub_prev, sub_input), axis = 0)
                self.gradients["b"] += (1 / m) * np.sum(np.squeeze(sub_input), axis = 0,
                                                        keepdims = True).T

        # remove the padding
        output = output_pad[:, ho : -ho, wo : -wo, :]

        # print("[INFO] backprop output shape at layer {}: {}".format(self.name, output.shape))
        # print("[INFO] Mean of backprop gradient at layer {}: {}".format(self.name, np.mean(output)))
        return output


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
        # get the height and width from the layer's input shape attribute
        (h, w) = self.input_shape[:2]

        # get the kernel dimensions from the layer's kernel size attribute
        (kh, kw) = self.kernel_size

        # get the stride dimension from the layer's stride attribute
        (sh, sw) = self.stride

        # calculate the output dimensions
        oh = (h - kh) // sh + 1
        ow = (w - kw) // sw + 1
        od = self.input_shape[2]

        # set the layer's output shape
        self.output_shape = (oh, ow, od)

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

    def forward_call(self, input):
        # calculate the number of training examples
        m = input.shape[0]

        # initialize the output tensor
        output = np.zeros((m, self.output_shape[0], self.output_shape[1], self.output_shape[2]))

        # loop over the output
        for i in range(0, output.shape[1]):
            for j in range(0, output.shape[2]):
                # define the sub input limits
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                # extract the sub input
                sub_input = input[:, h_start : h_end, w_start : w_end, :]

                # calculate the output value
                output[:, i, j, :] = np.amax(sub_input, axis = (1, 2))

        # initialize the layer cache
        self.cache = input

        # return the calculated value
        return output

    def __create_mask(self, arr):
        return (arr == np.max(arr))

    def backward_call(self, input):
        # calculate the number of training examples
        m = input.shape[0]

        # retrieve the information from the cache
        A_prev = self.cache

        # initialize the output tensor
        output = np.zeros((m, self.input_shape[0], self.input_shape[1], self.input_shape[2]))

        # loop over the number of training examples
        for e in range(m):
            for i in range(0, input.shape[1]):
                for j in range(0, input.shape[2]):
                    # calculate the sub_prev indices
                    h_start = i * self.stride[0]
                    h_end = h_start + self.kernel_size[0]
                    w_start = j * self.stride[1]
                    w_end = w_start + self.kernel_size[1]

                    # extract the sub prev and sub input
                    sub_prev = A_prev[:, h_start:h_end, w_start:w_end, :]
                    sub_input = input[:, i, j, :]

                    # expand the sub input to suit multiplication
                    sub_input = sub_input[:, np.newaxis, np.newaxis, :]

                    # get the mask for the sub region
                    mask = self.__create_mask(sub_prev)

                    # calculate the gradients
                    output[:, h_start:h_end, w_start:w_end, :] += np.multiply(
                        mask, sub_input)

        # return the calculated value
        # print("[INFO] backprop output shape at layer {}: {}".format(self.name, output.shape))
        # print("[INFO] Mean of backprop gradient at layer {}: {}".format(self.name, np.mean(output)))
        return output