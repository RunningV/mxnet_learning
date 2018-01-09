from mxnet import nd
from mxnet.gluon import nn

def get_net():
	net = nn.Sequential()
	with net.name_scope():
	  net.add(nn.Dense(4, activation="relu"))
	  net.add(nn.Dense(2))

	return net

x = nd.random.uniform(shape=(3, 5))

net = get_net()
net.initialize()
print(net(x))

w = net[0].weight
b = net[0].bias
print(net[0])
print(net[0].name)
print(w.data())
print(b.data())
