from tensorflow.keras import datasets
from torch.nn import Module
from tensorflow.python.keras import Model

import tensorflow as tf


class Trainer():
    def __inti__(self, net=None, dataset=None):
        self.net = None
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        if net:
            self.net = net
        if dataset:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self._load(dataset.lower())
        

    def train(self, net=None, dataset=None, x_train=None, y_train=None, batch_size=32, epochs=1):
        if net:
            self.net = net
        if dataset:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self._load(dataset.lower())
        if x_train and y_train:
            x, y = x_train, y_train
        else:
            x, y = self.x_train, self.y_train

        if isinstance(self.net, Module):
            self._train_torch_model(self.net, x, y, batch_size, epochs)

        if isinstance(self.net, Model):
            self._train_keras_model(self.net, x, y, batch_size, epochs)

    def evaluate(self, x_test=None, y_test=None):
        if x_test and y_test:
            x, y = x_test, y_test
        else:
            x, y = self.x_test, self.y_test
        if isinstance(self.net, Module):
            self._evaluate_torch_model(self.net, x, y)

        if isinstance(self.net, Model):
            self._evaluate_keras_model(self.net, x, y)

    def _train_torch_model(self, net, x_train, y_train, batch_size, epochs):
        pass

    def _train_keras_model(self, net, x_train, y_train, batch_size, epochs):
        history = net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    
    def _evaluate_torch_model(self, net, x, y):
        pass

    def _evaluate_keras_model(self, net, x, y):
        (loss, acc) = net.evaluate(x, y)
        print(f'loss: {loss}')
        print(f' acc: {acc}')

    def _load(self, dataset):
        print('loading: '+dataset)
        match dataset:
            case 'mnist':
                (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
                return (x_train.reshape(-1, 28, 28, 1)/255, tf.one_hot(y_train, 10)), (x_test.reshape(-1, 28, 28, 1)/255, tf.one_hot(y_test, 10))
            case 'fashion_mnist':
                (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
                return (x_train.reshape(-1, 28, 28, 1)/255, tf.one_hot(y_train, 10)), (x_test.reshape(-1, 28, 28, 1)/255, tf.one_hot(y_test, 10))
            case 'cifar10':
                (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
                return (x_train.reshape(-1, 28, 28, 3)/255, tf.one_hot(y_train, 10)), (x_test.reshape(-1, 28, 28, 3)/255, tf.one_hot(y_test, 10))
