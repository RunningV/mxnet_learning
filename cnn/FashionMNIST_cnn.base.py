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

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

weight_scale = 0.01

w1 = nd.random_normal(shape=(20, 1, 5, 5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(w1.shape[0], ctx=ctx)

w2 = nd.random_normal(shape=(50, 20, 3, 3), scale=weight_scale, ctx=ctx)
b2 = nd.zeros(w2.shape[0], ctx=ctx)

w3 = nd.random_normal(shape=(1250, 128), scale=weight_scale, ctx=ctx)
b3 = nd.zeros(w3.shape[1], ctx=ctx)

w4 = nd.random_normal(shape=(w3.shape[1], 10), scale=weight_scale, ctx=ctx)
b4 = nd.zeros(w4.shape[1], ctx=ctx)

params = [w1, b1, w2, b2, w3, b3, w4, b4]
for p in params:
	p.attach_grad()

def net(x, verbose=False):
	x = x.reshape((-1, 1, 28, 28)).as_in_context(ctx)
	h1_conv = nd.Convolution(data=x, weight=w1, bias=b1, kernel=w1.shape[2:], num_filter=w1.shape[0])
	h1_activation = nd.relu(h1_conv)
	h1 = nd.Pooling(data=h1_activation, pool_type='max', kernel=(2, 2), stride=(2, 2))

	h2_conv = nd.Convolution(data=h1, weight=w2, bias=b2, kernel=w2.shape[2:], num_filter=w2.shape[0])
	h2_activation = nd.relu(h2_conv)
	h2 = nd.Pooling(data=h2_activation, pool_type='max', kernel=(2, 2), stride=(2, 2))
	h2 = nd.flatten(h2)

	h3_linear = nd.dot(h2, w3) + b3
	h3 = nd.relu(h3_linear)

	h4_linear = nd.dot(h3, w4) + b4

	if verbose:
		print('1st conv block:', h1.shape)
		print('2nd conv block:', h2.shape)
		print('1st dense:', h3.shape)
		print('2nd dense:', h4_linear.shape)
		print('output:', h4_linear)
	return h4_linear.as_in_context(ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
learn_rate = 0.2

for epoch in range(5):
	train_loss = 0
	train_acc = 0
	for data, label in train_data:
		label = label.as_in_context(ctx)
		with autograd.record():
			output = net(data)
			print('output')
			print(output)
			loss = softmax_cross_entropy(output, label)
		loss.backward()
		utils.SGD(params, learn_rate/batch_size)
		train_loss += nd.mean(loss).asscalar()
		train_acc += utils.accuracy(output, label)

	test_acc = utils.evaluate_accuracy(test_data, net,ctx)
	print('Epoch %d. Loss: %f, Train acc %f, Test acc %f' % (epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))