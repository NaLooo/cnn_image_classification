from tensorflow.python.keras import Model, Sequential
from tensorflow import keras
from models import vgg_16
from trainer import Trainer

import numpy as np

# seq = Sequential()
# f = issubclass(Sequential, Model)

# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# x_train = np.expand_dims(x_train.astype(np.float32), -1) / 255
# print(x_train.shape)
# print(x_train.dtype)


# model = vgg_16((32,32,3), 10)
model = vgg_16((28,28,1), 10)
trainer = Trainer()

trainer.train(model, 'fashion_mnist', epochs=5)
# trainer.train(model, 'cifar10', epochs=5)
trainer.evaluate()

