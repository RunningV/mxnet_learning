from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
import matplotlib.pyplot as plt


def transform(data, label):
	return data.astype('float32')/255, label.astype('float32')

mnist_train = gluon.data.vision.FashionMNIST(root='../data/fashion-mnist', train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(root='../data/fashion-mnist', train=False, transform=transform)


data, label = mnist_train[0]
print(data.shape, label)
# (28, 28, 1) 2.0

def show_image(image):
	n = image.shape[0]
	_, figs = plt.subplots(1, n, figsize=(15, 15))
	for i in range(n):
		figs[i].imshow(image[i].reshape((28, 28)).asnumpy())
		figs[i].axes.get_xaxis().set_visible(False)
		figs[i].axes.get_yaxis().set_visible(False)
	plt.show()

def get_text_labels(label):
	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress,', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	return [text_labels[int(i)] for i in label]

data, label = mnist_train[0:9]
show_image(data)
print(label)
print(get_text_labels(label))

batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

num_inputs = 784
num_outputs = 10
w = nd.random_normal(shape=(num_inputs, num_outputs))
b = nd.random_normal(shape=(num_outputs))

params = [w, b]

for p in params:
	p.attach_grad()

def softmax(x):
	exp = nd.exp(x)
	part = exp.sum(axis=1, keepdims=True)
	return exp / part

x = nd.random_normal(shape=(2, 5))
x_prob = softmax(x)
print(x_prob)
print(x_prob.sum(axis=1))

def net(x):
	return softmax(nd.dot(x.reshape((-1, num_inputs)), w) + b)

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

loss = evaluate_accuracy(test_data)
print(loss)

def SGD(params, lr):
	for p in params:
		p[:] = p - lr*p.grad

learn_rate = 0.1

for epoch in range(5):
	train_loss = 0
	train_accuracy = 0
	for data, label in train_data:
		with autograd.record():
			output = net(data)
			loss = cross_entropy(output, label)
		loss.backward()
		SGD(params, learn_rate / batch_size)

		train_loss += nd.mean(loss).asscalar()
		train_accuracy += accuracy(output, label)

	test_accuracy = evaluate_accuracy(test_data)
	print('Epoch %d. Loss: %f, Train accuracy %f, Test accuracy %f'%(
		epoch, train_loss/len(train_data), train_accuracy/len(train_data), test_accuracy))

data, label = mnist_test[0:9]
show_image(data)
print('true labels:')
print(get_text_labels(label))

pred_labels = net(data).argmax(axis=1)
print('predicted labels:')
print(get_text_labels(pred_labels.asnumpy()))