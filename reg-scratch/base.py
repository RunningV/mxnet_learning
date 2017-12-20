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
def data_iter(num_examples):
	idx = list(range(num_examples))
	random.shuffle(idx)
	for i in range(0, num_examples, batch_size):
		j = nd.array(idx[i:min(i+batch_size, num_examples)])
		yield x.take(j), y.take(j)

def init_params():
	w = nd.random_normal(scale=1, shape=(num_inputs, 1))
	b = nd.zeros(shape=(1, ))
	params = [w, b]
	for p in params:
		p.attach_grad()
	return params

def L2_penalty(w, b):
	return ((w**2).sum() + b**2) / 2

def net(X, w, b):
    return nd.dot(X, w) + b

def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

def test(net, params, X, y):
    return square_loss(net(X, *params), y).mean().asscalar()
    #return np.mean(square_loss(net(X, *params), y).asnumpy())

def train(lambd):
    epochs = 10
    learning_rate = 0.005
    w, b = params = init_params()
    train_loss = []
    test_loss = []
    for e in range(epochs):
        for data, label in data_iter(num_train):
            with autograd.record():
                output = net(data, *params)
                loss = square_loss(
                    output, label) + lambd * L2_penalty(*params)
            loss.backward()
            sgd(params, learning_rate, batch_size)
        train_loss.append(test(net, params, X_train, y_train))
        test_loss.append(test(net, params, X_test, y_test))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.show()
    return 'learned w[:10]:', w[:10].T, 'learned b:', b

train(0)