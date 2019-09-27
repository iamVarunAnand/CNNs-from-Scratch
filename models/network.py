# import the necessary packages
from layers.core import *

class Node:
    """
    A node is the basic building block of the network (graph). Each node comprises of three
    attributes. A reference to the parent node, the layer object corresponding to the current
    node, and a reference to the next node.
    """

    def __init__(self, layer):
        # initialize a node(layer) in the graph
        self.prev = None
        self.layer = layer
        self.next = None

    def set_prev_layer(self, layer):
        # set the reference to the previous layer in the network
        self.prev = layer

    def set_next_layer(self, layer):
        # set the reference to the next layer in the network
        self.next = layer


class Network:
    """
    A network is a directed acyclic graph formed by connecting various Nodes according to the
    specified model architecture.
    """

    def __init__(self):
        # initialize the attributes of the network
        self.layer_list = []
        self.nodes = []

    def add_to_network(self, layer):
        # check if the current layer is Conv2D
        if layer.__class__.__name__ == "Conv2D":
            # initialize the pre-reshape layer
            pre_reshape_layer = PreConvReshape(filters = layer.filters, kernel_size =
            layer.kernel_size, stride = layer.stride, name = "pre_reshape_" + layer.name)

            # initialize the post-reshape layer
            post_reshape_layer = PostConvReshape(filters = layer.filters,
                                                 name = "post_reshape_" + layer.name)

            # add the pre reshape layer to the layer list
            self.layer_list.append(pre_reshape_layer)

            # add the convolutional layer to the layer list
            self.layer_list.append(layer)

            # add the post reshape layer to the layer list
            self.layer_list.append(post_reshape_layer)
        elif layer.__class__.__name__ == "MaxPooling2D":
            # initialize the pre-reshape layer
            pre_reshape_layer = PrePoolReshape(kernel_size = layer.kernel_size, stride =
            layer.kernel_size, name = "pre_reshape_" + layer.name)

            # initialize the post-reshape layer
            post_reshape_layer = PostPoolReshape(name = "post_reshape_" + layer.name)

            # add the pre reshape layer to the layer list
            self.layer_list.append(pre_reshape_layer)

            # add the pooling layer to the layer list
            self.layer_list.append(layer)

            # add the post reshape layer to the layer list
            self.layer_list.append(post_reshape_layer)
        else:
            # append the layer to the layer list
            self.layer_list.append(layer)

    def build_network(self):
        """
        Initializes the directed acyclic graph based on the layer objects in layer list
        """

        # loop over all the layers in the model and build the network
        for (i, layer) in enumerate(self.layer_list):
            if layer.__class__.__name__ == "Input":
                self.nodes.append(Node(layer))
            else:
                # initialize a node with the current layer
                node = Node(layer)

                # set a reference from the current node to the previous node
                node.set_prev_layer(self.nodes[i - 1])

                # set a reference from the previous node to the current node
                self.nodes[i - 1].set_next_layer(node)

                # add the node to the nodes list
                self.nodes.append(node)

    def init_network(self):
        # loop over all the nodes in the network and initialize them
        for node in self.nodes:
            try:
                node.layer.init_layer(input_shape = node.prev.layer.output_shape)
            except:
                node.layer.init_layer()

    def visualize_network(self):
        # loop over all the nodes in the network
        for node in self.nodes:
            print("[INFO] {} input_shape: {}, output_shape: {}".
                  format(node.layer.name, node.layer.input_shape, node.layer.output_shape))

    def forward_pass(self, x_train):
        # loop over all the nodes in the graph for the forward pass
        output = x_train
        for node in self.nodes:
            output = node.layer.forward_call(output)

        return output

    def backward_pass(self, input):
        # loop over all the nodes (except softmax) in the graph for the backward pass
        output = input
        for node in reversed(self.nodes[:-1]):
            output = node.layer.backward_call(output)

        return output

    def update_weights(self, lr):
        # loop over all the nodes in the graph to update the layer weights
        for node in self.nodes:
            node.layer.update_weights(lr)