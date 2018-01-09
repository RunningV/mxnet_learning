from mxnet.gluon import nn
from mxnet import nd
from mxnet import gluon
from mxnet import init
import mxnet
import utils

ctx = mxnet.gpu()

def vgg_block(num_convs, channels):
	out = nn.Sequential()
	for _ in range(num_convs):
		out.add(nn.Conv2D(channels=channels, kernel_size=3, padding=1, activation='relu'))
	out.add(nn.MaxPool2D(pool_size=2, strides=2))
	return out

def vgg_stack(arch):
	out = nn.Sequential()
	for (num_convs, channels) in arch:
		out.add(vgg_block(num_convs, channels))
	return out

block = vgg_block(2, 128)
block.initialize()
x = nd.random.uniform(shape=(2, 3, 16, 16))
y = block(x)

num_outputs = 10
arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
net = nn.Sequential()

with net.name_scope():
	net.add(
		vgg_stack(arch),
		nn.Flatten(),
		nn.Dense(4096, activation='relu'),
		nn.Dropout(0.5),
		nn.Dense(4096, activation='relu'),
		nn.Dropout(0.5),
		nn.Dense(num_outputs)
	)

net.initialize(ctx=ctx, init=init.Xavier())
train_data, test_data = utils.load_data_fashion_mnist(batch_size=10, resize=96)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})
utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)