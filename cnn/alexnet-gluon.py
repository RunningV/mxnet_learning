import mxnet as mx
import utils
from mxnet.gluon import nn
from mxnet import nd
from mxnet import init
from mxnet import gluon
ctx = mx.gpu()

net = nn.Sequential()
with net.name_scope():
		net.add(
		nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'),
		nn.MaxPool2D(pool_size=3, strides=2),

		nn.Conv2D(channels=256, kernel_size=5, padding=2, activation='relu'),
		nn.MaxPool2D(pool_size=3, strides=2),

		nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
		nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
		nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'),
		nn.MaxPool2D(pool_size=3, strides=2),

		nn.Flatten(),
		nn.Dense(4096, activation='relu'),
		nn.Dropout(0.5),

		nn.Dense(4096, activation='relu'),
		nn.Dropout(0.5),

		nn.Dense(10)
	)
net.initialize(ctx=ctx, init=init.Xavier())

batch_size=64
learn_rate = 0.6
train_data, test_data = utils.load_data_fashion_mnist(batch_size=10, resize=96)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})
utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=3)