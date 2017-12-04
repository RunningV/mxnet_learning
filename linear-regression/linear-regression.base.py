from mxnet import ndarray as nd
from mxnet import autograd
import matplotlib.pyplot as plt
import random

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

x = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0]*x[:,0] + true_w[1]*x[:,1] + true_b
y += 0.01*nd.random_normal(shape=y.shape)

print(x[0], y[0])

# plt.scatter(x[:,1].asnumpy(), y.asnumpy())
# plt.show()

batch_size = 10
def data_iter():
	idx = list(range(num_examples)) 
	random.shuffle(idx)
	for i in range(0, num_examples, batch_size):
		j = nd.array(idx[i:min(i+batch_size, num_examples)])
		yield nd.take(x, j), nd.take(y, j)

for data, label in data_iter():
	print(data, label)
	break

# 初始化模型参数
w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1,))
params = [w, b]

for p in params:
	p.attach_grad()

def net(x):
	return nd.dot(x, w) + b

def square_loss(yhat, y):
	return (yhat - y.reshape(yhat.shape))**2

def SGD(params, lr):
	for p in params:
		p[:] = p - lr*p.grad

def real_fn(x):
	return true_w[0]*x[:,0] + true_w[1]*x[:,1] + true_b

def plot(losses, x, sample_size=100):
	xs = list(range(len(losses)))
	f, (fg1, fg2) = plt.subplots(1,2)
	fg1.set_title('Loss during training')
	fg1.plot(xs, losses, '-r')
	fg2.set_title('Estimated vs real function')
	fg2.plot(x[:sample_size, 1].asnumpy(), net(x[:sample_size, :]).asnumpy(), 'or', label='Estimated')
	fg2.plot(x[:sample_size, 1].asnumpy(), real_fn(x[:sample_size, :]).asnumpy(), '*g', label='Real')

	fg2.legend()
	plt.show()

epochs = 5
learn_rate = 0.001
niter = 0
losses = []
moving_loss = 0
smoothing_constant = 0.01

for e in range(epochs):
	total_loss = 0
	for data, label in data_iter():
		with autograd.record():
			output = net(data)
			loss = square_loss(output, label)
		loss.backward()
		SGD(params, learn_rate)
		total_loss += nd.sum(loss).asscalar()

		niter += 1
		curr_loss = nd.mean(loss).asscalar()
		moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss

		est_loss = moving_loss / (1 - (1 - smoothing_constant)**niter)

		if(niter+1)%100:
			losses.append(est_loss)
			print('Epoch %s, batch %s, Moving avg of loss: %s, Average loss: %f'%(e, niter, est_loss, total_loss/num_examples))
			plot(losses, x)