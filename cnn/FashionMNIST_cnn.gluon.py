import mxnet as mx

ctx = mx.gpu()
# try:
# 	ctx = mx.gpu()
# 	_ = nd.zeros((1,), ctx=ctx)
# except:
# 	ctx = mx.cpu()

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
import matplotlib.pyplot as plt
import utils

net = gluon.nn.Sequential()
with net.name_scope():
	net.add(
		gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
		gluon.nn.MaxPool2D(pool_size=2, strides=2),
		gluon.nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
		gluon.nn.MaxPool2D(pool_size=2, strides=2),
		gluon.nn.Flatten(),
		gluon.nn.Dense(128, activation='relu'),
		gluon.nn.Dense(10)
	)
net.initialize(ctx=ctx) 

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

utils.train(train_data, test_data, net, loss, trainer, ctx, batch_size, epochs=5)