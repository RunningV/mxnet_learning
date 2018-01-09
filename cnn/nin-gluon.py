from mxnet.gluon import nn
from mxnet import nd
from mxnet import gluon
from mxnet import init
import utils
import mxnet

ctx = mxnet.gpu()

def mlpconv(channels, kernel_size, padding, strides=1, max_pooling=True):
	out = nn.Sequential()
	out.add(
		nn.Conv2D(channels=channels, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu'),
		nn.Conv2D(channels=channels, kernel_size=1, strides=1, padding=0, activation='relu'),
		nn.Conv2D(channels=channels, kernel_size=1, strides=1, padding=0, activation='relu')		
	)
	if max_pooling:
		out.add(nn.MaxPool2D(pool_size=3, strides=2))
	return out

block = mlpconv(64, 3, 0)
block.initialize()
x = nd.random.uniform(shape=(32, 3, 16, 16))
y = block(x)
print(y.shape)

net = nn.Sequential()
with net.name_scope():
	net.add(
		mlpconv(96, 11, 0, strides=4),
		mlpconv(256, 5, 2),
		mlpconv(384, 3, 1),
		nn.Dropout(0.5),
		mlpconv(10, 3, 1, max_pooling=False),
		nn.AvgPool2D(pool_size=5),
		nn.Flatten()
	)

net.initialize(ctx=ctx, init=init.Xavier())
train_data, test_data = utils.load_data_fashion_mnist(batch_size=10, resize=224)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})
utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)
