# import the necessary packages
<<<<<<< HEAD
from sklearn.preprocessing import LabelBinarizer
from keras.datasets import mnist
=======
import numpy as np

>>>>>>> refs/remotes/origin/master
from project.models import Model
from project.layers.convolutional import Conv2D, MaxPooling2D
from project.layers.core import Input
from project.layers.core import Dense
from project.layers.core import Flatten
<<<<<<< HEAD
from project.layers.activation import ReLU, Softmax
import numpy as np

# set the seed for the random number generator
np.random.seed(5)

def build_model():
    model = Model()
    model.add(Input(input_shape = (128, 28, 28, 1)))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), stride = (1, 1), padding_type = "same"))
    model.add(ReLU())
    model.add(MaxPooling2D(kernel_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(units = 32))
    model.add(ReLU())
    model.add(Dense(units = 64))
    model.add(ReLU())
    model.add(Dense(units = 10))
    model.add(Softmax())

    # return the constructed architecture
    return model

# load the dataset
((x_train, y_train), (x_test, y_test)) = mnist.load_data()

# normalize the images into the range [0, 1]
x_train = x_train.astype("float") / 255.0
x_train = np.expand_dims(x_train, axis = -1)
x_test = x_test.astype("float") / 255.0
x_test = np.expand_dims(x_test, axis = -1)

# convert the labels from integers into vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# initialize the model
model = build_model()
model.compile(loss = "categorical_crossentropy", lr = 0.1, metrics = ["loss", "accuracy"])

# train the model
model.fit(x_train[:1024], y_train[:1024], epochs = 30, batch_size = 128)
=======
from project.layers.activation import ReLU


model = Model()
model.add(Input(input_shape = (28, 28, 1)))
model.add(Conv2D(filters = 32, kernel_size = (3, 3), stride = (1, 1), padding_type = "same"))
model.add(MaxPooling2D(kernel_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 32))
model.add(ReLU())
model.add(Dense(units = 64))
model.add(Dense(units = 10))
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# initialize a random input
x_train = np.random.randn(28, 28, 1)
y_train = np.random.randn(10, 1)
model.fit(x_train, y_train)
>>>>>>> refs/remotes/origin/master
