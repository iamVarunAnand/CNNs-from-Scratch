# import the necessary packages
from .layer import Layer
<<<<<<< HEAD
import multiprocessing as mp
=======
>>>>>>> refs/remotes/origin/master
import numpy as np


class Conv2D(Layer):
    """
    Implements the convolutional layer. Filters are applied to the input tensor which is then
    transformed into the appropriate output tensor.
    """
<<<<<<< HEAD
    def __init__(self, filters, kernel_size, stride = (1, 1), padding_type = "same", name = None):
        # call the parent classZ constructor
=======
    def __init__(self, filters, kernel_size, stride, padding_type = "same", name = None):
        # call the parent class constructor
>>>>>>> refs/remotes/origin/master
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
<<<<<<< HEAD
        # set the layer's output shape
        self.output_shape = (self.input_shape[0], self.input_shape[1], self.filters)
=======
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
>>>>>>> refs/remotes/origin/master

    def init_layer(self, input_shape = None):
        # set the layer's input_shape
        self.set_input_shape(input_shape)

<<<<<<< HEAD
        # print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight and bias matrices
        self.weights = np.random.randn(self.filters, self.kernel_size[0], self.kernel_size[1],
                                       self.input_shape[-1]) * 0.01
        self.weights = self.weights[np.newaxis, : ,: , :, :].astype(np.float16)
        self.bias = np.zeros((self.filters, 1), dtype = np.float16)
=======
        print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight and bias matrices
        self.weights = np.random.randn(self.filters, self.kernel_size[0], self.kernel_size[1],
                                       self.input_shape[-1])
        self.bias = np.zeros((self.filters, 1))
>>>>>>> refs/remotes/origin/master

        # set the layer's output shape
        self.set_output_shape()

        # set the layer's gradients
        self.set_gradients()

<<<<<<< HEAD
        # print(self.name, "output_shape = ", self.output_shape, sep = " ")
=======
        print(self.name, "output_shape = ", self.output_shape, sep = " ")
>>>>>>> refs/remotes/origin/master

    def __pad_input(self, input):
        (hp, wp) = self.padding

<<<<<<< HEAD
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

        # # extract the sub prev
        # sub_prev = A_prev[:, h_start : h_end, w_start : w_end, :]
        #
        # # reshape and expand the sub prev to suit multiplication
        # sub_prev = sub_prev[:, np.newaxis, :, :, :]
        #
        # self.gradients["W"] += (1 / m) * np.sum(np.multiply(sub_prev, sub_input), axis = 0)
        # self.gradients["b"] += (1 / m) * np.sum(np.squeeze(sub_input), axis = 0,
        #                                         keepdims = True).T
        #
        # # remove the padding
        # output = output_pad[:, ho : -ho, wo : -wo, :]

        # print("[INFO] backprop output shape at layer {}: {}".format(self.name, output.shape))
        # print("[INFO] Mean of backprop gradient at layer {}: {}".format(self.name, np.mean(output)))
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
=======
        return np.pad(input, pad_width = [(hp, hp), (wp, wp), (0, 0)], mode = "constant",
                      constant_values = (0, 0))

    def __convolve(self, arr_1, arr_2):
        return np.sum(np.multiply(arr_1, arr_2))

    def forward_call(self, input):
        # initialize the output tensor
        output = np.zeros(self.output_shape)

        # pad the input based on the padding attribute
        if self.padding is not None:
            input = self.__pad_input(input)

        # loop over the filters
        for f in range(self.filters):
            # output height index
            oh = 0
            for i in range(self.padding[0], input.shape[0] - 1, self.stride[0]):
                # output width index
                ow = 0
                for j in range(self.padding[1], input.shape[1] - 1, self.stride[1]):
                    # extract the individual offset values
                    (ho, wo) = self.offset

                    # extract the sub input from the input based on the offsets
                    sub_input = input[i - ho:i + ho + 1, j - ho:j + ho + 1, :]

                    # convolve the image and the filter and add the bias
                    output[oh, ow, f] = np.add(self.__convolve(self.weights[f], sub_input),
                                               self.bias[f])


                    # increment the output width index
                    ow = ow + 1
                # increment the output height index
                oh = oh + 1

        # initialize the layer's cache
        self.cache = input

        # return the calculated value
        return output

    def backward_call(self, input):
        # retrieve the information from the cache
        A_prev = self.cache

        # initialize the output
        output = np.zeros(self.input_shape)

        # apply padding to A_prev and the output
        A_prev = self.__pad_input(A_prev)
        output = self.__pad_input(output)

        # loop over the filters
        for f in range(self.filters):
            for i in range(self.padding[0], input.shape[0], self.stride[0]):
                for j in range(self.padding[1], input.shape[1], self.stride[1]):
                    # extract the individual offset values
                    (ho, wo) = self.offset

                    # extract the sub input from the input based on the offsets
                    sub_prev = A_prev[i - ho:i + ho + 1, j - ho:j + ho + 1, :]

                    # calculate the gradients
                    output[i - ho:i + ho + 1, j - ho:j + ho + 1, :] += np.multiply(self.weights[f],
                                                                                   input[i, j, f])

                    self.gradients["W"][f] += sub_prev * input[i, j, f]
                    self.gradients["b"][f] += input[i, j, f]

        # return the calculated value after removing the padding
        (ho, wo) = self.offset
        print("[INFO] backprop output shape at layer {}: {}".format(self.name, output[ho : -ho, wo : -wo].shape))
        return output[ho : -ho, wo : -wo]

>>>>>>> refs/remotes/origin/master

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
<<<<<<< HEAD
        # set the layer's output shape
        self.output_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[-1])
=======
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
>>>>>>> refs/remotes/origin/master

    def init_layer(self, input_shape = None):
        # set the layer's input shape
        self.set_input_shape(input_shape)

<<<<<<< HEAD
        # print(self.name, "input_shape = ", self.input_shape, sep = " ")
=======
        print(self.name, "input_shape = ", self.input_shape, sep = " ")
>>>>>>> refs/remotes/origin/master

        # initialize the weight and bias matrices
        self.weights = None
        self.bias = None

        # set the layer's output shape
        self.set_output_shape()

        # set the layer's gradients
        self.set_gradients()

<<<<<<< HEAD
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
=======
        print(self.name, "output_shape = ", self.output_shape, sep = " ")

    def forward_call(self, input):
        # initialize the output tensor
        output = np.zeros(self.output_shape)

        # output height index
        oh = 0
        for i in range(self.offset[0], input.shape[0] - 1, self.stride[0]):
            # output width index
            ow = 0
            for j in range(self.offset[1], input.shape[1] - 1, self.stride[1]):
                # extract the individual offset values
                (ho, wo) = self.offset

                # extra the sub input from the input based on the offsets
                sub_input = input[i - ho:i + ho, j - ho:j + ho, :]

                # find the maximum value in the sub_input
                output[oh, ow, :] = np.amax(sub_input, axis = (0, 1))

                # increment the output width index
                ow = ow + 1

            # increment the output height index
            oh = oh + 1

        # initialize the layer cache
        self.cache = input

        # return the calculated value
        return output
>>>>>>> refs/remotes/origin/master

    def __create_mask(self, arr):
        return (arr == np.max(arr))

<<<<<<< HEAD
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
=======
    def backward_call(self, input):
        # retrieve the information from the cache
        A_prev = self.cache

        # initialize the output tensor
        output = np.zeros(self.input_shape)

        # loop over the number of filters
        for f in range(input.shape[-1]):
            for i in range(0, input.shape[0]):
                for j in range(0, input.shape[1]):
                    # calculate the sub_prev indices
                    h_start = i * self.stride[0]
                    h_end = h_start + self.kernel_size[0]
                    w_start = j * self.stride[1]
                    w_end = w_start + self.kernel_size[1]

                    # extract the sub prev region
                    sub_prev = A_prev[h_start:h_end, w_start:w_end, f]

                    # get the mask for the sub region
                    mask = self.__create_mask(sub_prev)

                    # calculate the gradients
                    output[h_start:h_end, w_start:w_end, f] += np.multiply(mask, input[i, j, f])

        # return the calculated value
        print("[INFO] backprop output shape at layer {}: {}".format(self.name, output.shape))
        return output
>>>>>>> refs/remotes/origin/master
