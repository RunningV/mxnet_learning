import utils
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn

ctx = mx.gpu()

net = nn.Sequential()
with net.name_scope():
	net.add(nn.Conv2D(channels=20, kernel_size=5))
	net.add(nn.BatchNorm(axis=1))
	net.add(nn.Activation(activation='relu'))
	net.add(nn.MaxPool2D(pool_size=2, strides=2))

	net.add(nn.Conv2D(channels=50, kernel_size=3))
	net.add(nn.BatchNorm(axis=1))
	net.add(nn.Activation(activation='relu'))
	net.add(nn.MaxPool2D(pool_size=2, strides=2))
	net.add(nn.Flatten())

	net.add(nn.Dense(128, activation='relu'))

	net.add(nn.Dense(10))

net.initialize(ctx=ctx)
batch_size = 256
learn_rate = 0.6
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer =  gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learn_rate})

for epoch in range(5):
	train_loss = 0
	train_acc = 0
	for data, label in train_data:
		data = data.reshape((-1, 1, 28, 28)).as_in_context(ctx)
		label = label.as_in_context(ctx)
		with autograd.record():
			output = net(data)
			loss = softmax_cross_entropy(output, label)
		loss.backward()
		trainer.step(batch_size)
		train_loss += nd.mean(loss).asscalar()
		train_acc += utils.accuracy(output, label)

	test_acc = utils.evaluate_accuracy(test_data, net, ctx)
	print('Epoch %d. Loss: %f, Train acc %f, Test acc %f' % (epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))