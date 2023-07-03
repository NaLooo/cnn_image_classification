from tensorflow.keras import datasets
from torch.nn import Module
from tensorflow.python.keras import Model
from tqdm import trange
from torch.optim import Adam
import torch

import torch.nn.functional as F
import tensorflow as tf
import numpy as np


class Trainer():
    def __init__(self, net=None, dataset=None):
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
        net.train()
        opt = Adam(net.parameters(), 1e-3)
        x_train = torch.from_numpy(np.array(x_train, dtype=np.float32).transpose(0,3,1,2))
        y_train = torch.from_numpy(np.array(y_train, dtype=np.float32))

        for e in range(epochs):
            permutation = np.random.permutation(x_train.shape[0])
            x_train = x_train[permutation]
            y_train = y_train[permutation]

            for i in trange(0, x_train.shape[0], batch_size):

                images = x_train[i:i+batch_size]
                labels = y_train[i:i+batch_size]

                output = net(images)

                loss = F.cross_entropy(output, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def _train_keras_model(self, net, x_train, y_train, batch_size, epochs):
        history = net.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    
    def _evaluate_torch_model(self, net, x, y):
        print('evaluating...')
        net.eval()
        x = torch.from_numpy(np.array(x, dtype=np.float32).transpose(0,3,1,2))
        y = torch.from_numpy(np.array(y, dtype=np.float32))
        correct = 0
        list = []
        for i in trange(x.shape[0]):
            image = x[i].reshape(1, *x[i].shape)
            label = torch.argmax(y[i])

            output = net(image)
            if torch.argmax(output) == label:
                correct += 1
            else:
                list.append(i)

        print('test accuracy: %.4f' % (correct/y.shape[0]))

    def _evaluate_keras_model(self, net, x, y):
        print('evaluating...')
        (loss, acc) = net.evaluate(x, y)
        print('loss: %.4f' % loss)
        print(' acc: %.4f' % acc)

    def _load(self, dataset: str):
        print('loading: ' + dataset)
        match dataset.lower():
            case 'mnist':
                (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
                return (x_train.reshape(-1, 28, 28, 1)/255, tf.one_hot(y_train, 10)), (x_test.reshape(-1, 28, 28, 1)/255, tf.one_hot(y_test, 10))
            case 'fashion_mnist':
                (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
                return (x_train.reshape(-1, 28, 28, 1)/255, tf.one_hot(y_train, 10)), (x_test.reshape(-1, 28, 28, 1)/255, tf.one_hot(y_test, 10))
            case 'cifar10':
                (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
                return (x_train.reshape(-1, 32, 32, 3)/255, tf.one_hot(y_train.reshape(-1), 10)), (x_test.reshape(-1, 32, 32, 3)/255, tf.one_hot(y_test.reshape(-1), 10))
            case 'cifar100':
                (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
                return (x_train.reshape(-1, 32, 32, 3)/255, tf.one_hot(y_train.reshape(-1), 100)), (x_test.reshape(-1, 32, 32, 3)/255, tf.one_hot(y_test.reshape(-1), 100))
            
