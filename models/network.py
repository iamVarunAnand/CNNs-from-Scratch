# import the necessary packages


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
        # append the layer to the layer list
        self.layer_list.append(layer)

    def build_network(self):
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