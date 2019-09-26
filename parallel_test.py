# import the necessary packages
import multiprocessing as mp
import numpy as np
import time
import concurrent.futures

# # ========================= multiple training examples - forward =========================
# m = 64
#
# input_shape = (m, 5, 5, 3)
# weights_shape = (4, 3, 3, 3)
# output_shape = (m, input_shape[1], input_shape[2], weights_shape[0])
#
# # initialize the tensors
# input_tensor = np.ones(input_shape)
# weight_tensor = np.random.randint(0, 10, size = weights_shape) * 0.1
# output_tensor = np.zeros(output_shape)
#
# # pad the input tensor
# input_tensor = np.pad(input_tensor, [(0, 0), (1, 1), (1, 1), (0, 0)], mode = "constant",
#                       constant_values = (0, 0))
#
# # expand the weight tensor for multiplication
# weight_tensor = weight_tensor[np.newaxis, :, :, :, :]
#
# for i in range(1, output_shape[1] + 1):
#     for j in range(1, output_shape[2] + 1):
#         # define the sub input limits
#         h_start = i - 1
#         h_end = i + 1
#         w_start = j - 1
#         w_end = j + 1
#
#         # extract the sub input
#         sub_input = input_tensor[:, h_start : h_end + 1, w_start : w_end + 1]
#
#         # expand the sub input tensor for multiplication
#         sub_input = sub_input[:, np.newaxis, :, :, :]
#
#         # calculate the output value
#         output_tensor[:, i - 1, j - 1, :] = np.sum(np.multiply(sub_input, weight_tensor),
#                                                 axis = (-1, -2, -3))

# # ========================= multiple training examples - forward =========================

m = 1

input_shape = (m, 5, 5, 1)
weights_shape = (4, 3, 3, 1)
output_shape = (m, input_shape[1], input_shape[2], weights_shape[0])

# initialize the tensors
input_tensor = np.ones(input_shape)
weight_tensor = np.random.randint(0, 10, size = weights_shape) * 0.1
# output_tensor = np.zeros(output_shape)

# expand the weight tensor for multiplication
weight_tensor = weight_tensor[np.newaxis, :, :, :, :]

def multiple_forward(input):
    output = np.sum(np.multiply(input, weight_tensor), axis = (-1, -2, -3))

    return output

def extract_sub_inputs():
    # pad the input tensor
    input_tensor_ = np.pad(input_tensor, [(0, 0), (1, 1), (1, 1), (0, 0)], mode = "constant",
                          constant_values = (0, 0))

    shape_2 = input_tensor_.shape[1] - 3 + 1
    shape_3 = input_tensor_.shape[2] - 3 + 1
    shape_0 = shape_2 * shape_3
    sub_input_shape = (shape_0, m, 3, 3, 1)

    sub_inputs = np.empty(sub_input_shape)

    for i in range(0, shape_2):
        for j in range(0, shape_3):
            # define the sub input limits
            h_start = i
            h_end = i + 3
            w_start = j
            w_end = j + 3

            sub_inputs[i * 3 + j] = input_tensor_[:, h_start : h_end, w_start : w_end, :]

    return sub_inputs

sub_inputs = extract_sub_inputs()

# sub_inputs = sub_inputs[:, :, np.newaxis, :, :]
#
# start_time = time.time()
#
# pool = mp.Pool(8)
# result = pool.map(multiple_forward, sub_inputs)
#
# end_time = time.time()
#
# result = np.swapaxes(np.array(result), 0, 1)
# result = np.reshape(result, (result.shape[0], 5, 5, 4))
#
# print("[INFO] shape of the result = {}".format(result.shape))
# print("[INFO] time taken = {}".format(end_time - start_time))
#
# # checking result
# print(result[0, :, :, 0])

# print(result[0].shape)

# for i in range(1, output_shape[1] + 1):
#     for j in range(1, output_shape[2] + 1):
#         # define the sub input limits
#         h_start = i - 1
#         h_end = i + 1
#         w_start = j - 1
#         w_end = j + 1
#
#         result = pool.map(multiple_forward, input_tensor[:, h_star])