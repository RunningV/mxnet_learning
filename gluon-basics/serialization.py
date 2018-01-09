from mxnet import nd
from mxnet.gluon import nn

x = nd.ones(3)
y = nd.zeros(4)

# 保存模型数据
filename = './data/test1.params'
nd.save(filename, [x, y])
# 读取模型参数
a, b = nd.load(filename)
print(a, b)

def get_net():
	net = nn.Sequential()
	with net.name_scope():
	  net.add(nn.Dense(10, activation="relu"))
	  net.add(nn.Dense(5))

	return net

net = get_net()
net.initialize()
x = nd.random.uniform(shape=(2, 10))
print(x)
print(net(x))
'''
通过get_net()将x(2,10)变为x(2, 5),中间两层dense做隐藏层
'''

filename = './data/mpl.params'
net.save_params(filename)