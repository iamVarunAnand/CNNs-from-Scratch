# import the necessary packages
import numpy as np

from project.models import Model
from project.layers.convolutional import Conv2D, MaxPooling2D
from project.layers.core import Input
from project.layers.core import Dense
from project.layers.core import Flatten
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