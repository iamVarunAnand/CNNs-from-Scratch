from keras.applications import VGG16
from keras.applications import InceptionV3
from keras.applications import ResNet50
from keras.utils import plot_model

vgg = VGG16()
inception = InceptionV3()
resnet = ResNet50()

plot_model(vgg, to_file = "output/visualizations/VGG16.png", show_shapes = True)
plot_model(inception, to_file = "output/visualizations/InceptionV3.png", show_shapes = True)
plot_model(resnet, to_file = "output/visualizations/ResNet50.png", show_shapes = True)
