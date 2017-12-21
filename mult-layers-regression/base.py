from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
import matplotlib.pyplot as plt


def transform(data, label):
	return data.astype('float32')/255, label.astype('float32')

mnist_train = gluon.data.vision.FashionMNIST(root='../data/fashion-mnist', train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(root='../data/fashion-mnist', train=False, transform=transform)

batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

num_inputs = 28 * 28
num_outputs = 10

num_hidden = 256
num_hidden10 = 512
weight_scale1 = 0.2
weight_scale10 = 0.3
weight_scale2 = 0.5

w1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale1)
b1 = nd.zeros(num_hidden)

# w10 = nd.random_normal(shape=(num_hidden, num_hidden10), scale=weight_scale10)
# b10 = nd.zeros(num_hidden10)

w2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale2)
b2 = nd.zeros(num_outputs)

params = [w1, b1, w2, b2]

for p in params:
	p.attach_grad()

def relu(x):
 	return nd.maximum(x, 0)

def net(x):
 	x = x.reshape((-1, num_inputs))
 	h1 = relu(nd.dot(x, w1) + b1)
 	# h10 = relu(nd.dot(h1, w10) + b10)
 	output = nd.dot(h1, w2) + b2
 	return output

def cross_entropy(y_, y):
	return - nd.pick(nd.log(y_), y)

def accuracy(output, label):
	return nd.mean(output.argmax(axis=1)==label).asscalar()

def evaluate_accuracy(data_iter):
	acc = 0
	for data, label in data_iter:
		output = net(data)
		acc += accuracy(output, label)
	return acc / len(data_iter)

def SGD(params, lr):
	for p in params:
		p[:] = p - lr*p.grad

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learn_rate = 0.5

for epoch in range(5):
	train_loss = 0
	train_accuracy = 0
	for data, label in train_data:
		with autograd.record():
			output = net(data)
			loss = softmax_cross_entropy(output, label)
		loss.backward()
		SGD(params, learn_rate/batch_size)
		train_loss += nd.mean(loss).asscalar()
		train_accuracy += accuracy(output, label)

	test_accuracy = evaluate_accuracy(test_data)
	print('Epoch %d. Loss: %f, Train accuracy %f, Test accuracy %f'%(
		epoch, train_loss/len(train_data), train_accuracy/len(train_data), test_accuracy))




