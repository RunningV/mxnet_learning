from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.dpi'] = 120

num_train = 100
num_test = 100
true_w = [1.2, -3.4, 5.6]
true_b = 5.0

x0 = nd.random.normal(shape=(num_train + num_test, 1))
x = nd.concat(x0, nd.power(x0, 2), nd.power(x0, 3))
y = true_w[0] * x[:, 0] + true_w[1] * x[:, 1] + true_w[2] * x[:, 2] + true_b
y += 0.1 * nd.random.normal(shape=y.shape)

def train(x_train, x_test, y_train, y_test):
	net = gluon.nn.Sequential()
	with net.name_scope():
		net.add(gluon.nn.Dense())
	net.initialize()

	learn_rate = 0.01
	epochs = 100
	batch_size = min(10, y_train.shape[0])

	dateset_train = gluon.data.ArrayDataset(x_train, y_train)
	data_iter_train = gluon.data.DataLoader(dateset_train, batch_size, shuffle=True)

	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learnning_rate': learn_rate})
	square_loss = gluon.loss.L2Loss()

	train_loss = []
	test_loss = []
	for e in range(epochs):
		for data, label in data_iter_train:
			with autograd.record():
				output = net(data)
				loss = square_loss(output, label)
			loss.backward()
			trainer.step(batch_size)
		train_loss.append(square_loss(net(x_train), y_train).mean().asscalar())
		test_loss.append(square_loss(net(x_test), y_test).mean().asscalar())

	plt.plot(train_loss)
	plt.plot(test_loss)
	plt.legend(['train', 'test'])
	plt.show()
	return ('learned weight', net[0].weight.data(), 'learned bias', net[0].bias.data())

train(x[: num_train, :], x[num_train:, :], y[:num_train], y[num_train:])

