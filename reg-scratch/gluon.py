from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import mxnet
import random
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.dpi'] = 120


num_train = 20
num_test = 100
num_inputs = 200

true_w = nd.ones((num_inputs, 1)) * 0.01
true_b = 0.05

x = nd.random.normal(shape=(num_train+num_test, num_inputs))
y = nd.dot(x, true_w)
y += 0.01*nd.random.normal(shape=y.shape)
x_train, x_test = x[:num_train, :], x[num_train:, :]
y_train, y_test = y[:num_train], y[num_train:]

batch_size = 1
dataset_train = gluon.data.ArrayDataset(x_train, y_train)
data_iter = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True)

square_loss = gluon.loss.L2Loss()


def test(net, X, y):
    return square_loss(net(X), y).mean().asscalar()

def train(weight_decay):
    epochs = 10
    learning_rate = 0.005
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.collect_params().initialize(mxnet.init.Normal(sigma=1))

    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': weight_decay})

    train_loss = []
    test_loss = []
    for e in range(epochs):
        for data, label in data_iter:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
        train_loss.append(test(net, x_train, y_train))
        test_loss.append(test(net, x_test, y_test))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.show()
    print('learned w[:10]:', net[0].weight.data()[:,:10], 'learned b:', net[0].bias.data())

train(5)
