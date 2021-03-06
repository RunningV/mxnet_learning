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
		net.add(gluon.nn.Dense(1))
	net.initialize()

	learn_rate = 0.01
	epochs = 100
	batch_size = min(10, y_train.shape[0])

	dateset_train = gluon.data.ArrayDataset(x_train, y_train)
	data_iter_train = gluon.data.DataLoader(dateset_train, batch_size, shuffle=True)

	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learn_rate})
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
	print ('learned weight', net[0].weight.data(), 'learned bias', net[0].bias.data())

# 使用线性拟合x0容易发生欠拟合：20次后训练值loss趋于0，但测试值loss趋于100
train(x0[: num_train, :], x0[num_train:, :], y[:num_train], y[num_train:])
# 使用与目标函数对应的三阶多项式100次的训练正常: 20次后训练值、测试值的loss都趋于0
train(x[: num_train, :], x[num_train:, :], y[:num_train], y[num_train:])
# 使用与目标函数对应的三阶多项式10次的训练过拟合：训练值loss趋于0，但测试值loss依然很高
train(x[: 10, :], x[num_train:, :], y[:10], y[num_train:])
#正常时用loss衡量模型的表现，训练值loss与测试值loss的曲线基本是一致的
#过拟合或欠拟合时，训练值的loss可能都趋于0，但测试值的loss都比较高；过拟合或欠拟合的本旨并不能通过loss值来表达。
#本例的做法实际是：通过给出一系列的x, y值，求y与x的关系。当我们用x0线性拟合方程时得不到好的关系模型
#因为实际模型是y = w1*x + w2*x*x + w3*x*x*x + b, 我们实际需要求w1、w2、w3、b
#而求解的原理是通过微积分求导寻找loss趋于0时的系数。

