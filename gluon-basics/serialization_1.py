import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

def get_net():
	net = nn.Sequential()
	with net.name_scope():
	  net.add(nn.Dense(10, activation="relu"))
	  net.add(nn.Dense(5))

	return net

x = nd.ones(shape=(2, 10))
print(x)
filename = './data/mpl.params'
net = get_net()
net.load_params(filename, mx.cpu())
print(net(x))