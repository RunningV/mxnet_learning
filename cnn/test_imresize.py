from mxnet import gluon
from mxnet import ndarray as nd
import matplotlib.pyplot as plt
import utils
import mxnet
ctx = mxnet.gpu()

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')

mnist_train = gluon.data.vision.FashionMNIST(root='../data/fashion-mnist', train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(root='../data/fashion-mnist', train=True, transform=None)

def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()

data, label = mnist_train[:2]
print(data.shape)
print(label)
show_images(data)

train_data = utils.DataLoader(mnist_test, 2, shuffle=True, resize=224)

for i, batch in enumerate(train_data):
	if i < 2:
		data1, label1 = utils._get_batch(batch, ctx)
		print(data1.shape)
		print(label1)
		show_images(data1)
