import mxnet as mx
import utils
from mxnet import autograd
from mxnet import gluon
from mxnet import ndarray as nd
ctx = mx.gpu()

weight_scale = 0.01

c1 = 20
w1 = nd.random.normal(shape=(c1, 1, 5, 5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(c1, ctx=ctx)

gamma1 = nd.random.normal(shape=c1, scale=weight_scale, ctx=ctx)
beta1 = nd.random.normal(shape=c1, scale=weight_scale, ctx=ctx)
moving_mean1 = nd.zeros(c1, ctx=ctx)
moving_variance1 = nd.zeros(c1, ctx=ctx)

c2 = 50
w2 = nd.random.normal(shape=(c2, c1, 3, 3), scale=weight_scale, ctx=ctx)
b2 = nd.zeros(c2, ctx=ctx)

gamma2 = nd.random.normal(shape=c2, scale=weight_scale, ctx=ctx)
beta2 = nd.random.normal(shape=c2, scale=weight_scale, ctx=ctx)
moving_mean2 = nd.zeros(c2, ctx=ctx)
moving_variance2 = nd.zeros(c2, ctx=ctx)

c3 = 128
w3 = nd.random.normal(shape=(1250, c3), scale=weight_scale, ctx=ctx)
b3 = nd.zeros(c3, ctx=ctx)

w4 = nd.random.normal(shape=(w3.shape[1], 10), scale=weight_scale, ctx=ctx)
b4 = nd.zeros(10, ctx=ctx)

params = [w1, b1, gamma1, beta1, w2, b2, gamma2, beta2, w3, b3, w4, b4]
for p in params:
	p.attach_grad()

def net(x, is_training=False, verbose=False):
	x = x.as_in_context(w1.context)
	h1_conv = nd.Convolution(data=x, weight=w1, bias=b1, kernel=w1.shape[2:], num_filter=c1)
	h1_bn = utils.batch_norm(h1_conv, gamma1, beta1, is_training, moving_mean1, moving_variance1)
	h1_activation = nd.relu(h1_conv)
	h1 = nd.Pooling(data=h1_activation, pool_type='max', kernel=(2, 2), stride=(2, 2))

	h2_conv = nd.Convolution(data=h1, weight=w2, bias=b2, kernel=w2.shape[2:], num_filter=c2)
	h2_bn = utils.batch_norm(h2_conv, gamma2, beta2, is_training, moving_mean2, moving_variance2)
	h2_activation = nd.relu(h2_conv)
	h2 = nd.Pooling(data=h2_activation, pool_type='max', kernel=(2, 2), stride=(2, 2))
	h2 = nd.flatten(h2)

	h3_linear = nd.dot(h2, w3) + b3
	h3 = nd.relu(h3_linear)

	h4_linear = nd.dot(h3, w4) + b4

	if verbose:
		print('h1 conv block: ', h1.shape)
		print('h2 conv block: ', h2.shape)
		print('h3 conv block: ', h3.shape)
		print('h4 conv block: ', h4_linear.shape)
		print('output: ', h4_linear)

	return h4_linear.as_in_context(ctx)

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
learn_rate = 0.2

for epoch in range(5):
	train_loss = 0
	train_acc = 0
	for data, label in train_data:
		data = data.reshape((-1, 1, 28, 28)).as_in_context(ctx)
		label = label.as_in_context(ctx)
		with autograd.record():
			output = net(data, is_training=True)
			loss = softmax_cross_entropy(output, label)
		loss.backward()
		utils.SGD(params, learn_rate/batch_size)
		train_loss += nd.mean(loss).asscalar()
		train_acc += utils.accuracy(output, label)

	test_acc = utils.evaluate_accuracy(test_data, net, ctx)
	print('Epoch %d. Loss: %f, Train acc %f, Test acc %f' % (epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))