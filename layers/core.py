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
        super(Input, self).__init__(name)

        # set the input shape for the input layer
        self.input_shape = input_shape

    def set_output_shape(self):
        # set the output shape of the input layer equal to the input shape
        self.output_shape = self.input_shape

    def init_layer(self, input_shape = None):
<<<<<<< HEAD
        # print(self.name, "input_shape = ", self.input_shape, sep = " ")
=======
        print(self.name, "input_shape = ", self.input_shape, sep = " ")
>>>>>>> refs/remotes/origin/master

        # initialize the weight and bias matrices
        self.weights = None
        self.bias = None

        # set the output shapes
        self.set_output_shape()

        # set the layer's gradients
        self.set_gradients()

<<<<<<< HEAD
        # print(self.name, "output_shape = ", self.output_shape, sep = " ")
=======
        print(self.name, "output_shape = ", self.output_shape, sep = " ")
>>>>>>> refs/remotes/origin/master

    def forward_call(self, input):
        # return the input unchanged
        return input

    def backward_call(self, input):
        # return the input unchanged (provided only for completeness)
        return input

class Dense(Layer):
    """
    The Dense layer represents the fully connected block. Input tensor is multiplied by the
    layer's weight matrix and the output tensor is returned.
    """
    def __init__(self, units, name = None):
        # call the parent class constructor
        super(Dense, self).__init__(name)
        self.units = units

    def set_output_shape(self):
        # set the layer's output shape
<<<<<<< HEAD
        self.output_shape = (self.input_shape[0], self.units, 1)
=======
        self.output_shape = (self.units, 1)
>>>>>>> refs/remotes/origin/master

    def init_layer(self, input_shape = None):
        # set the input shapes
        self.set_input_shape(input_shape = input_shape)

<<<<<<< HEAD
        # print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight and bias matrices
        self.weights = np.random.randn(self.units, self.input_shape[1]) * 0.1
        self.weights = self.weights.astype(np.float16)

        self.bias = np.zeros((self.units, ), dtype = np.float16)
=======
        print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight and bias matrices
        self.weights = np.random.randn(self.units, self.input_shape[0])
        self.bias = np.zeros((self.units, 1))
>>>>>>> refs/remotes/origin/master

        # set the output shapes
        self.set_output_shape()

        # set the layer's gradients
        self.set_gradients()

<<<<<<< HEAD
        # print(self.name, "output_shape = ", self.output_shape, sep = " ")

    def forward_call(self, input):
        # calculate the output
        output = np.add(np.dot(input, self.weights.T), self.bias)
=======
        print(self.name, "output_shape = ", self.output_shape, sep = " ")

    def forward_call(self, input):
        # calculate the output
        output = np.add(np.dot(self.weights, input), self.bias)
>>>>>>> refs/remotes/origin/master

        # initialize the layer's cache
        self.cache = (input, output)

        # return the calculated value
        return output

    def backward_call(self, input):
        # extract the values from the cache
        (A_prev, z) = self.cache

<<<<<<< HEAD
        # calculate the total numbers of examples
        m = input.shape[0]

        # calculate the gradients
        output = np.dot(input, self.weights)

        self.gradients["W"] = (1 / m) * np.dot(input.T, A_prev)
        self.gradients["b"] = (1 / m) * np.sum(input, axis = 0, keepdims = True)

        # return the calculated value
        # print("[INFO] backprop output shape at layer {}: {}".format(self.name, output.shape))
        # print("[INFO] Mean of backprop gradient at layer {}: {}".format(self.name, np.mean(output)))
        return output

=======
        # calculate the gradients
        output = np.dot(self.weights.T, input)
        self.gradients["W"] = np.dot(input, A_prev.T)
        self.gradients["b"] = input

        # return the calculated value
        print("[INFO] backprop output shape at layer {}: {}".format(self.name, output.shape))
        return output


>>>>>>> refs/remotes/origin/master
class Flatten(Layer):
    """
    Flattens the input along the axis specified.
    """
    def __init__(self, name = None):
        # call the parent class constructor
        super(Flatten, self).__init__(name)

    def set_output_shape(self):
        # calculate the total number of elements in the input tensor
        total_elts = 1
<<<<<<< HEAD
        for elts in self.input_shape[1:]:
            total_elts = total_elts * elts

        # set the layer's output shape
        self.output_shape = (self.input_shape[0], total_elts)
=======
        for elts in self.input_shape:
            total_elts = total_elts * elts

        # set the layer's output shape
        self.output_shape = (total_elts, 1)
>>>>>>> refs/remotes/origin/master

    def init_layer(self, input_shape = None):
        # set the input shapes
        self.set_input_shape(input_shape)

<<<<<<< HEAD
        # print(self.name, "input_shape = ", self.input_shape, sep = " ")
=======
        print(self.name, "input_shape = ", self.input_shape, sep = " ")
>>>>>>> refs/remotes/origin/master

        # initialize the weight and bias matrices
        self.weights = None
        self.bias = None

        # set the output shapes
        self.set_output_shape()

        # set the layer's gradients
        self.set_gradients()

<<<<<<< HEAD
        # print(self.name, "output_shape = ", self.output_shape, sep = " ")

    def forward_call(self, input):
        # calculate the number of training examples
        m = input.shape[0]

        # initialize the output tensor
        output = np.reshape(input, self.output_shape)

        # return the calculated value
        return output

    def backward_call(self, input):
        # calculate the number of training examples
        m = input.shape[0]

        # initialize the output tensor
        output = np.reshape(input, self.input_shape)

        # return the calculated value
        # print("[INFO] backprop output shape at layer {}: {}".format(self.name, output.shape))
        # print("[INFO] Mean of backprop gradient at layer {}: {}".format(self.name, np.mean(output)))
        return output

class PreConvReshape(Layer):
    def __init__(self, filters, kernel_size, stride = (1, 1), name = None):
        # call the parent class constructor
        super(PreConvReshape, self).__init__(name)

        # initialize the layer attributes
        self.units = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

    def set_output_shape(self):
        # set the layer's output shape
        self.output_shape = (self.input_shape[1] * self.input_shape[2], self.input_shape[0],
                             self.kernel_size[0], self.kernel_size[1], self.input_shape[3])

    def init_layer(self, input_shape = None):
        # set the layer's input shape
        self.set_input_shape(input_shape)

        # print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the layer's weights
        self.weights = None
        self.bias = None

        # set the layer's output shape
        self.set_output_shape()

        # print(self.name, "output_shape = ", self.output_shape, sep = " ")

        # initialize the layer's gradients
        self.set_gradients()

    def __pad_input(self, input):
        (hp, wp) = self.padding

        return np.pad(input, pad_width = [(0, 0), (hp, hp), (wp, wp), (0, 0)], mode = "constant",
                      constant_values = (0, 0))

    def __get_sub_input_index(self, i, j):
        # convert the 2D index into 1D index
        return (i * self.input_shape[1] + j)

    def forward_call(self, input):
        # pad the input with zeros
        input = self.__pad_input(input)

        # initialize the output tensor
        output = np.zeros(self.output_shape, dtype = np.float16)

        for i in range(0, self.input_shape[1]):
            for j in range(0, self.input_shape[2]):
                # calculate the sub input limits
                h_start = i
                h_end = i * self.stride[0] + self.kernel_size[0]
                w_start = j
                w_end = j * self.stride[1] + self.kernel_size[1]

                # extract the sub input
                sub_input = input[:, h_start : h_end, w_start : w_end, :]

                # initialize the corresponding output
                output[self.__get_sub_input_index(i, j)] = sub_input

        # return the calculated value
        return output

    def backward_call(self, input):
        # initialize the output tensor
        output = np.zeros(self.input_shape, dtype = np.float16)

        # pad the output
        output = self.__pad_input(output)

        for i in range(0, self.input_shape[1]):
            for j in range(0, self.input_shape[2]):
                # calculate the sub input limits
                h_start = i
                h_end = i * self.stride[0] + self.kernel_size[0]
                w_start = j
                w_end = j * self.stride[1] + self.kernel_size[1]

                # calculate the output value
                output[:, h_start : h_end, w_start : w_end, :] += input[self.__get_sub_input_index(i, j)]

        # return the calculated value
        (hp, wp) = self.padding
        return output[:, hp : -hp, wp : -wp, :]

class PostConvReshape(Layer):
    def __init__(self, filters, name = None):
        # call the parent class constructor
        super(PostConvReshape, self).__init__(name)

        # intialize the layer attributes
        self.filters = filters

    def set_output_shape(self):
        # set the layer's output shape
        self.output_shape = (self.input_shape[1], int(np.sqrt(self.input_shape[0])), int(np.sqrt(
            self.input_shape[0])), self.filters)

    def init_layer(self, input_shape = None):
        # set the layer's input shape
        self.set_input_shape(input_shape)

        # print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the layer's weights
        self.weights = None
        self.bias = None

        # set the layer's output shape
        self.set_output_shape()

        # print(self.name, "output_shape = ", self.output_shape, sep = " ")

        # initialize the layer's gradients
        self.set_gradients()

    def forward_call(self, input):
        # swap the first two axes in the input
        np.swapaxes(input, 0, 1)

        # reshape the 1D output of the convolutional layer into 2D output
=======
        print(self.name, "output_shape = ", self.output_shape, sep = " ")

    def forward_call(self, input):
        # reshape the input tensor
>>>>>>> refs/remotes/origin/master
        output = np.reshape(input, self.output_shape)

        # return the calculated value
        return output

    def backward_call(self, input):
<<<<<<< HEAD
        # swap the first two axes in the input
        np.swapaxes(input, 0, 1)

        # reshape the 1D output of the convolutional layer into 2D output
        output = np.reshape(input, self.input_shape)

        # return the calculated value
        return output

class PrePoolReshape(Layer):
    def __init__(self, kernel_size, stride, name = None):
        # call the parent class constructor
        super(PrePoolReshape, self).__init__(name)

        # set the layer's attributes
        self.kernel_size = kernel_size
        self.stride = stride

    def set_output_shape(self):
        # get the height and width from the layer's input shape attribute
        (h, w) = self.input_shape[1:3]

        # get the kernel dimensions from the layer's kernel size attribute
        (kh, kw) = self.kernel_size

        # get the stride dimension from the layer's stride attribute
        (sh, sw) = self.stride

        # calculate the output dimensions
        oh = (h - kh) // sh + 1
        ow = (w - kw) // sw + 1

        # set the layer's output shape
        self.output_shape = (oh * ow, self.input_shape[0], kh, kw, self.input_shape[-1])

    def init_layer(self, input_shape = None):
        # set the layer's input shape
        self.set_input_shape(input_shape)

        # print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight matrices
        self.weights = None
        self.bias = None

        # set the layer's output shape
        self.set_output_shape()

        # print(self.name, "output_shape = ", self.output_shape, sep = " ")

        # initialize the layer's gradients
        self.set_gradients()

    def __get_sub_input_index(self, i, j):
        n_cols = int(np.sqrt(self.output_shape[0]))
        return(i * n_cols + j)

    def forward_call(self, input):
        # initialize the output tensor
        output = np.zeros(self.output_shape, dtype = np.float16)

        for i in range(0, self.input_shape[1] // self.kernel_size[0]):
            for j in range(0, self.input_shape[2] // self.kernel_size[1]):
                # calculate the sub input limits
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                # extract the sub input
                sub_input = input[:, h_start: h_end, w_start: w_end, :]

                # initialize the corresponding output
                output[self.__get_sub_input_index(i, j)] = sub_input

        # return the calculated value
        return output

    def backward_call(self, input):
        # initialize the output tensor
        output = np.zeros(self.input_shape, dtype = np.float16)

        for i in range(0, self.input_shape[1] // self.kernel_size[0]):
            for j in range(0, self.input_shape[2] // self.kernel_size[0]):
                # calculate the sub input limits
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                # calculate the sub output
                output[:, h_start: h_end, w_start: w_end, :] += input[self.__get_sub_input_index(i, j)]

        # return the calculated value
        return output

class PostPoolReshape(Layer):
    def __init__(self, name = None):
        # call the parent class constructor
        super(PostPoolReshape, self).__init__(name)

    def set_output_shape(self):
        # calculate the height and width of the feature maps
        h = int(np.sqrt(self.input_shape[0]))
        w = int(np.sqrt(self.input_shape[0]))

        # calculate the other dimensions
        m = self.input_shape[1]
        d = self.input_shape[-1]

        # set the layer's output shape
        self.output_shape = (m, h, w, d)

    def init_layer(self, input_shape = None):
        # set the layer's input shape
        self.set_input_shape(input_shape)

        # print(self.name, "input_shape = ", self.input_shape, sep = " ")

        # initialize the weight matrices
        self.weights = None
        self.bias = None

        # set the layer's output shape
        self.set_output_shape()

        # print(self.name, "output_shape = ", self.output_shape, sep = " ")

        # initialize the layer's gradients
        self.set_gradients()

    def forward_call(self, input):
        # swap the first and second axes
        np.swapaxes(input, 0, 1)

        # calculate the output value
        output = np.reshape(input, self.output_shape)

        # return the calculated value
        return output

    def backward_call(self, input):
        # swap the first and second axes
        np.swapaxes(input, 0, 1)

        # calculate the output value
        output = np.reshape(input, self.input_shape)

        # return the calculated value
        return output
=======
        # reshape the input tensor
        output = np.reshape(input, self.input_shape)

        # return the calculated value
        print("[INFO] backprop output shape at layer {}: {}".format(self.name, output.shape))
        return output

>>>>>>> refs/remotes/origin/master
