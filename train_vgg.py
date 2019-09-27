# import the necessary packages
# from keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from conv import VGG16
import numpy as np

# # load the dataset
# ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
#
# # apply padding, and normalize the images into the range [0, 1]
# x_train = x_train[:, :, :, np.newaxis]
# x_train = np.pad(x_train, [(0, 0), (2, 2), (2, 2), (0, 0)], mode = "constant",
#                  constant_values = [(0, 0)])
# x_train = x_train.astype(np.float32) / 255.0
#
# # convert the labels from integers into vectors
# lb = LabelBinarizer()
# y_train = lb.fit_transform(y_train)

# initialize the model
model = VGG16.build(32, 32, 1, 10, batch_size = 128)
model.compile(loss = "categorical_crossentropy", lr = 0.01, metrics = ["loss", "accuracy"])
model.summary()

# # train the model
# model.fit(x_train[:1024], y_train[:1024], batch_size = 128, epochs = 2)