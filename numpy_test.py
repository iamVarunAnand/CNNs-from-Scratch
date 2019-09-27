# import the necessary packages
import numpy as np

# set the random number generator seed
np.random.seed(7)

# # ========================= single training example - forward =========================
# input_shape = (5, 5, 3)
# weights_shape = (4, 3, 3, 3)
# output_shape = (input_shape[0], input_shape[1], weights_shape[0])
#
# # initialize the tensors
# input_tensor = np.ones(input_shape)
# weight_tensor = np.random.randint(0, 10, size = weights_shape) * 0.1
# output_tensor = np.zeros(output_shape)
#
# # pad the input tensor
# input_tensor = np.pad(input_tensor, [(1, 1), (1, 1), (0, 0)], mode = "constant",
#                       constant_values = (0, 0))
#
# for i in range(1, output_shape[0] + 1):
#     for j in range(1, output_shape[1] + 1):
#         # define the sub input limits
#         h_start = i - 1
#         h_end = i + 1
#         w_start = j - 1
#         w_end = j + 1
#
#         # extract the sub input
#         sub_input = input_tensor[h_start : h_end + 1, w_start : w_end + 1]
#
#         # calculate the output value
#         output_tensor[i - 1, j - 1, :] = np.sum(np.multiply(sub_input, weight_tensor),
#                                                 axis = (-1, -2, -3))
#
# # ========================= single training example - forward =========================

# ========================= multiple training examples - forward =========================
m = 128

input_shape = (m, 28, 28, 1)
weights_shape = (64, 3, 3, 1)
output_shape = (m, input_shape[1], input_shape[2], weights_shape[0])

# initialize the tensors
input_tensor = np.ones(input_shape)
weight_tensor = np.random.randint(0, 10, size = weights_shape) * 0.1
output_tensor = np.zeros(output_shape)

# pad the input tensor
input_tensor = np.pad(input_tensor, [(0, 0), (1, 1), (1, 1), (0, 0)], mode = "constant",
                      constant_values = (0, 0))

# expand the weight tensor for multiplication
weight_tensor = weight_tensor[np.newaxis, :, :, :, :]

for i in range(1, output_shape[1] + 1):
    for j in range(1, output_shape[2] + 1):
        # define the sub input limits
        h_start = i - 1
        h_end = i + 1
        w_start = j - 1
        w_end = j + 1

        # extract the sub input
        sub_input = input_tensor[:, h_start : h_end + 1, w_start : w_end + 1]

        # expand the sub input tensor for multiplication
        sub_input = sub_input[:, np.newaxis, :, :, :]

        # calculate the output value
        output_tensor[:, i - 1, j - 1, :] = np.sum(np.multiply(sub_input, weight_tensor),
                                                axis = (-1, -2, -3))

# check the calculations
print(input_tensor[0, 0, 0, :])

print(weight_tensor[0, 0, :, :, 0])
print(weight_tensor[0, 1, :, :, 0])
print(weight_tensor[0, 2, :, :, 0])
print(weight_tensor[0, 3, :, :, 0])

print(output_tensor[0, :, :, 0])

# # ========================= multiple training examples - forward =========================

# # ========================= single training example - backward =========================
# input_shape = (5, 5, 3)
# weights_shape = (4, 3, 3, 3)
# output_shape = (input_shape[0], input_shape[1], weights_shape[0])
#
# # initialize the tensors
# input_tensor = np.zeros(input_shape)
# weight_tensor = np.random.randint(0, 10, size = weights_shape) * 0.1
# output_tensor = np.ones(output_shape)
#
# # pad the input tensor
# input_tensor = np.pad(input_tensor, [(1, 1), (1, 1), (0, 0)], mode = "constant",
#                       constant_values = (0, 0))
#
# for i in range(1, output_shape[0] + 1):
#     for j in range(1, output_shape[1] + 1):
#         # define the sub input limits
#         h_start = i - 1
#         h_end = i + 1
#         w_start = j - 1
#         w_end = j + 1
#
#         # # extract the sub input
#         # sub_input = input_tensor[h_start : h_end + 1, w_start : w_end + 1]
#
#         # extract the sub output
#         sub_output = output_tensor[i - 1, j - 1, :]
#
#         # reshape and expand the sub output to suit multiplication
#         sub_output = sub_output.T[:, np.newaxis, np.newaxis, np.newaxis]
#
#         # calculate the gradient
#         input_tensor[h_start: h_end + 1, w_start: w_end + 1, :] += np.sum(
#             np.multiply(sub_output, weight_tensor), axis = 0)
#
# input_tensor = input_tensor[1:-1, 1:-1, :]
#
#
# # check the calculations
# print(output_tensor[0, 0, :])
#
# print(weight_tensor[0, :, :, 0])
# print(weight_tensor[1, :, :, 0])
# print(weight_tensor[2, :, :, 0])
# print(weight_tensor[3, :, :, 0])
#
# print(input_tensor[:, :, 0])
# # ========================= single training example - backward =========================

# # ========================= multiple training examples - backward =========================
# m = 128
#
# input_shape = (m, 5, 5, 3)
# weights_shape = (4, 3, 3, 3)
# output_shape = (m, input_shape[1], input_shape[2], weights_shape[0])
#
# # initialize the tensors
# input_tensor = np.zeros(input_shape)
# weight_tensor = np.random.randint(0, 10, size = weights_shape) * 0.1
# output_tensor = np.ones(output_shape)
#
# # pad the input tensor
# input_tensor = np.pad(input_tensor, [(0, 0), (1, 1), (1, 1), (0, 0)], mode = "constant",
#                       constant_values = (0, 0))
#
# # expand the weight tensor to suit multiplication
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
#         # extract the sub output
#         sub_output = output_tensor[:, i - 1, j - 1, :]
#
#         # reshape and expand the sub output to suit multiplication
#         sub_output = sub_output[:, :, np.newaxis, np.newaxis, np.newaxis]
#
#         # calculate the gradient value
#         input_tensor[:, h_start: h_end + 1, w_start: w_end + 1, :] += (1 / m) * np.sum(
#             np.multiply(sub_output, weight_tensor), axis = 1)
#
# input_tensor = input_tensor[:, 1:-1, 1:-1, :]
#
# # check the calculations
# print(output_tensor[0, 0, 0, :])
#
# print(weight_tensor[0, 0, :, :, 0])
# print(weight_tensor[0, 1, :, :, 0])
# print(weight_tensor[0, 2, :, :, 0])
# print(weight_tensor[0, 3, :, :, 0])
#
# print(input_tensor[0, :, :, 0])
#
# # ========================= multiple training examples - backward =========================
