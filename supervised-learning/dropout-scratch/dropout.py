'''
丢弃法的概念
在现代神经网络中，我们所指的丢弃法，通常是对输入层或者隐含层做以下操作：

随机选择一部分该层的输出作为丢弃元素；
把丢弃元素乘以0；
把非丢弃元素拉伸。
'''
from mxnet import nd

def dropout(x, drou_prop):
	keep_prop = 1- drou_prop
	assert 0 <= keep_prop <= 1

	if keep_prop == 0:
		return x.zeros_like()

	mask = nd.random.uniform(0, 1.0, x.shape, ctx=x.context) < keep_prop
	scale = 1/ keep_prop
	print(mask * x * scale)

A = nd.arange(20).reshape((5, 4))
dropout(A, 0)
dropout(A, 0.5)
dropout(A, 1)