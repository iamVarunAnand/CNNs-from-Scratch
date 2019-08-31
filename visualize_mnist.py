# import the necessary packages
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# load the data into memory
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print info regarding dataset
print("[INFO] training set size: {}".format(x_train.shape))
print("[INFO] image dimensions: {}".format(x_train.shape[1:]))

# plot 16 random images in the training dataset
indices = np.random.randint(0, x_train.shape[0], size = (16,))

# loop over the indices and plot each image
for (i, index) in enumerate(indices):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_train[index], cmap = "gray")

# save the plot to disk
OUTPUT_PATH = "output/plots/mnist_visualization.png"
plt.savefig(OUTPUT_PATH)
