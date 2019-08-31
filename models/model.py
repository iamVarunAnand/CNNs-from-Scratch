# import the necessary packages
from .network import Network
import numpy as np


class Model(Network):
    def __init__(self):
        # call the parent class constructor
        super().__init__()

        # initialize all model attributes
        self.metrics = {}
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        # append the layer object to layer list
        self.add_to_network(layer)

    def compile(self, loss, optimizer, metrics):
        # set the model loss and optimizer
        self.loss = loss
        self.optimizer = optimizer

        # build the network
        self.build_network()

        # initialize the network
        self.init_network()

        # initialize the metrics dictionary
        for metric_name in metrics:
            self.metrics[metric_name] = []

    # def fit(self):
    # def predict(self):
