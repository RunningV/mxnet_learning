from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
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
dataset = gluon.data.ArrayDataset(x, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

for data, label in data_iter:
	print(data, label)
	break

net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))
net.initialize()

square_loss = gluon.loss.L2Loss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

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
batch_size = 10

for e in range(epochs):
	total_loss = 0
	for data, label in data_iter:
		with autograd.record():
			output = net(data)
			loss = square_loss(output, label)
		loss.backward()
		trainer.step(batch_size)
		total_loss += nd.sum(loss).asscalar()
		print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))

