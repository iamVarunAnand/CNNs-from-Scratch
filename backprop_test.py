import numpy as np


def __convolve(arr_1, arr_2):
    return np.sum(np.multiply(arr_1, arr_2))

def __pad_input(input, padding = (2, 2)):
    (hp, wp) = padding

    return np.pad(input, pad_width = [(hp, hp), (wp, wp), (0, 0)], mode = "constant",
                  constant_values = (0, 0))

def forward_call(input, output_shape, weights, bias):
        # initialize the output tensor
        output = np.zeros(output_shape)

        input = __pad_input(input)

        # loop over the filters
        for f in range(output.shape[-1]):
            # output height index
            oh = 0
            for i in range(1, input.shape[0] - 1, 1):
                # output width index
                ow = 0
                for j in range(1, input.shape[1] - 1, 1):
                    print(i, j)
                #     # extract the individual offset values
                #     (ho, wo) = (1, 1)
                #
                #     # extract the sub input from the input based on the offsets
                #     sub_input = input[i - ho:i + ho, j - ho:j + ho, :]
                #
                #     # convolve the image and the filter and add the bias
                #     output[oh, ow, f] = np.add(__convolve(weights[f], sub_input), bias[f])
                #
                #     # increment the output width index
                #     ow = ow + 1
                # # increment the output height index
                # oh = oh + 1

        # initialize the layer's cache
        cache = input

        # return the calculated value
        return output, cache

def backward_call(input, cache, weights, gradients):
    # retrieve the information from the cache
    A_prev = cache

    # initialize the output
    output = np.zeros(A_prev.shape)

    # apply padding to A_prev and the output
    A_prev = __pad_input(A_prev)
    output = __pad_input(output)

    # loop over the filters
    for f in range(input.shape[-1]):
        for i in range(1, input.shape[0], 1):
            for j in range(1, input.shape[1], 1):
                # extract the individual offset values
                (ho, wo) = (1, 1)

                # extract the sub input from the input based on the offsets
                sub_prev = A_prev[i - ho:i + ho + 1, j - ho:j + ho + 1, :]

                # calculate the gradients
                output[i - ho:i + ho + 1, j - ho:j + ho + 1, :] += np.multiply(weights[f],
                                                                               input[i, j, f])

                gradients["W"][f] += sub_prev * input[i, j, f]
                gradients["b"][f] += input[i, j, f]

    return output[1:-1, 1:-1, :], gradients

np.random.seed(1)
input = np.random.randn(4, 4, 3)
W = np.random.randn(8, 2, 2, 3)
b = np.random.randn(8, 1)
gradients = {"W" : W, "b" : b}

output, cache = forward_call(input, (4, 4, 8), W, b)
output, gradients = backward_call(output, cache, W, gradients)

print(output.shape)