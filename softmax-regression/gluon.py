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
# show_image(data)
print(label)
print(get_text_labels(label))

batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

net = gluon.nn.Sequential()
with net.name_scope():
	net.add(gluon.nn.Flatten())
	net.add(gluon.nn.Dense(10))
net.initialize()

softmax_cross_entripy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.15})

def accuracy(output, label):
	return nd.mean(output.argmax(axis=1)==label).asscalar()

def evaluate_accuracy(data_iter):
	acc = 0
	for data, label in data_iter:
		output = net(data)
		acc += accuracy(output, label)
	return acc / len(data_iter)

learn_rate = 0.5

for epoch in range(5):
	train_loss = 0
	train_accuracy = 0
	for data, label in train_data:
		with autograd.record():
			output = net(data)
			loss = softmax_cross_entripy(output, label)
		loss.backward()
		trainer.step(batch_size)

		train_loss += nd.mean(loss).asscalar()
		train_accuracy += accuracy(output, label)

	test_accuracy = evaluate_accuracy(test_data)
	print('Epoch %d. Learn_rate: %f, Loss: %f, Train accuracy %f, Test accuracy %f'%(
		epoch, learn_rate, train_loss/len(train_data), train_accuracy/len(train_data), test_accuracy))
	if epoch == 0: learn_rate = 0.1
	if epoch == 1: learn_rate = 0.0001
	if epoch == 2: learn_rate = 0.000001
	if epoch == 3: learn_rate = 0.000000001


data, label = mnist_test[0:9]
# show_image(data)
print('true labels:')
print(get_text_labels(label))

pred_labels = net(data).argmax(axis=1)
print('predicted labels:')
print(get_text_labels(pred_labels.asnumpy()))