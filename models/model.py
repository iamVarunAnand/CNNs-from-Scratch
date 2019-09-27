# import the necessary packages
from .network import Network
import numpy as np
import progressbar

class Model(Network):
    def __init__(self):
        # call the parent class constructor
        super(Model, self).__init__()

        # initialize all model attributes
        self.metrics = {}
        self.loss = None
        self.lr = None

    def add(self, layer):
        # append the layer object to layer list
        self.add_to_network(layer)

    def compile(self, loss, lr, metrics):
        # set the model loss and optimizer
        self.loss = loss
        self.lr = lr

        # build the network
        self.build_network()

        # initialize the network
        self.init_network()

        # initialize the metrics dictionary
        for metric_name in metrics:
            self.metrics[metric_name] = []

    def summary(self):
        self.visualize_network()

    def calc_loss(self, y_true, y_pred):
        if self.loss == "categorical_crossentropy":
            return -np.sum(y_true * np.log(y_pred))

    def fit(self, x_train, y_train, epochs, batch_size):
        # loop over the epochs
        for epoch in range(epochs):
            # calculate the total number of examples
            m = x_train.shape[0]

            # initialize the progress bar
            widgets = ["Epoch {}: ".format(epoch + 1), progressbar.Percentage(), " ",
                       progressbar.Bar(), progressbar.ETA()]

            pbar = progressbar.ProgressBar(maxval = m // batch_size, widgets = widgets).start()

            # loop over the batches
            for (count, i) in enumerate(range(0, m, batch_size)):
                self.cost = 0

                # extract the current batch
                x_batch = x_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                # forward pass
                y_pred = self.forward_pass(x_batch)

                # # calculate the total cost
                self.cost = self.cost + self.calc_loss(y_batch, y_pred)

                # initialize the initial derivative
                dZ = y_pred - y_batch

                # backward pass
                bp_output = self.backward_pass(dZ)

                # update the layer weights
                self.update_weights(self.lr)

                # update the progressbar
                pbar.update(count)

                # if it is the last batch, print the loss
                if (i + batch_size) % m == 0:
                    self.metrics["loss"].append(self.cost)

            # close the progressbar
            pbar.finish()

            # print the metrics after the current epoch
            print("[INFO] loss after epoch {} = {}".format(epoch + 1, self.metrics["loss"][epoch]))



    # def predict(self):
