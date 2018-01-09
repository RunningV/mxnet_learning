import mxnet as mx

ctx = mx.gpu()

from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
import matplotlib.pyplot as plt


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

def load_data_fashion_mnist(batch_size, resize=None):

	def transform(data, label):
		if resize:
	    n = data.shape[0]
	    new_data = nd.zeros((n, resize, resize, data.shape[3]))
	    for i in range(n):
	      new_data[i] = image.imresize(data[i], resize, resize)
	    data = new_data
		return nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32')

	mnist_train = gluon.data.vision.FashionMNIST(root='../data/fashion-mnist', train=True, transform=transform)
	mnist_test = gluon.data.vision.FashionMNIST(root='../data/fashion-mnist', train=False, transform=transform)
	return gluon.data.DataLoader(mnist_train, batch_size, shuffle=True), gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

def softmax(x):
	exp = nd.exp(x)
	part = exp.sum(axis=1, keepdims=True)
	return exp / part

def cross_entropy(y_, y):
	return - nd.pick(nd.log(y_), y)

def accuracy(output, label):
	return nd.mean(output.argmax(axis=1)==label).asscalar()

def evaluate_accuracy(data_iter, net, ctx):
	acc = 0
	for data, label in data_iter:
		data = data.reshape((-1, 1, 28, 28)).as_in_context(ctx)
		label = label.as_in_context(ctx)
		output = net(data)
		acc += accuracy(output, label)
	return acc / len(data_iter)

def SGD(params, lr):
	for p in params:
		p[:] = p - lr*p.grad

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def train(train_data, test_data, net, lossFn, trainer, ctx, batch_size, epochs=5):
	for epoch in range(epochs):
		train_loss = 0
		train_acc = 0
		for data, label in train_data:
			data = data.reshape((-1, 1, 28, 28)).as_in_context(ctx)
			label = label.as_in_context(ctx)
			with autograd.record():
				output = net(data)
				loss = lossFn(output, label)
			loss.backward()
			trainer.step(batch_size)
			train_loss += nd.mean(loss).asscalar()
			train_acc += accuracy(output, label)

		test_acc = evaluate_accuracy(test_data, net)
		print('Epoch %d. Loss: %f, Train acc %f, Test acc %f' % (epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))

# 批量归一化
def batch_norm(X, gamma, beta, is_training, moving_mean, moving_variance,
               eps = 1e-5, moving_momentum = 0.9):
    assert len(X.shape) in (2, 4)
    # 全连接: batch_size x feature
    if len(X.shape) == 2:
        # 每个输入维度在样本上的平均和方差
        mean = X.mean(axis=0)
        variance = ((X - mean)**2).mean(axis=0)
    # 2D卷积: batch_size x channel x height x width
    else:
        # 对每个通道算均值和方差，需要保持4D形状使得可以正确的广播
        mean = X.mean(axis=(0,2,3), keepdims=True)
        variance = ((X - mean)**2).mean(axis=(0,2,3), keepdims=True)
        # 变形使得可以正确的广播
        moving_mean = moving_mean.reshape(mean.shape)
        moving_variance = moving_variance.reshape(mean.shape)

    # 均一化
    if not is_training:
        X_hat = (X - mean) / nd.sqrt(variance + eps)
        #!!! 更新全局的均值和方差
        moving_mean[:] = moving_momentum * moving_mean + (
            1.0 - moving_momentum) * mean
        moving_variance[:] = moving_momentum * moving_variance + (
            1.0 - moving_momentum) * variance
    else:
        #!!! 测试阶段使用全局的均值和方差
        X_hat = (X - moving_mean) / nd.sqrt(moving_variance + eps)

    # 拉升和偏移
    return gamma.reshape(mean.shape) * X_hat + beta.reshape(mean.shape)