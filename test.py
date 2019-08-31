# import the necessary packages
from project.models import Model
from project.layers.core import Input
from project.layers.core import Dense

model = Model()
model.add(Input(input_shape = (28, 28)))
model.add(Dense(units = 32))
model.add(Dense(units = 64))
model.add(Dense(units = 10))

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])