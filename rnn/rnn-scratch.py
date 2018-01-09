import random
import mxnet as mx
from mxnet import nd

ctx = mx.gpu()

with open('../data/jaychou_lyrics.txt', 'r', encoding='UTF-8') as f:
	words = f.read()
print(words[0:49])
print(len(words))

words = words.replace('\n', ' ').replace('\r', ' ')
train_words = words[0: 20000]
test_words = words[20000:]

idx_to_char = list(set(train_words))
char_to_idx = dict([char, i] for i, char in enumerate(idx_to_char))
vocab_size = len(char_to_idx)

words_indices = [char_to_idx[char] for char in train_words]

sample = words_indices[0:40]
print(''.join([idx_to_char[idx] for idx in sample]))
print(len(sample))

def data_iter_random(words_indices, batch_size, seq_len, ctx=None):
	num_example = (len(words_indices) - 1) // seq_len
	epoch_size = num_example // batch_size

	example_indices = list(range(num_example))
	random.shuffle(example_indices)

	def _data(pos):
		return words_indices[Pos: pos+seq_len]

	for i in range(epoch_size):
		i = i * batch_size
		batch_indices = example_indices[i:i + batch_size]
		data = nd.array([_data(j * seq_len) for j in batch_indices], ctx=ctx)
		label = nd.array([_data(j * seq_len+1) for j in batch_indices], ctx=ctx)
		yield data, label

def data_iter_consecutive(words_indices, batch_size, seq_len, ctx=None):
	words_indices = nd.array(words_indices, ctx=ctx)
	data_len = len(words_indices)
	batcn_len = data_len // batch_size

	indices = words_indices[0: batch_size * batcn_len].reshape((batch_size, batcn_len))
	epoch_size = (batcn_len - 1) // seq_len
	for i in range(epoch_size):
		i = i * seq_len
		data = indices[:, i: i + seq_len]
		label = indices[:, i + 1: i + seq_len + 1]
		yield data, label

def get_inputs(data):
  return [nd.one_hot(X, vocab_size) for X in data.T]

my_seq = list(range(30))

data, label = data_iter_consecutive(my_seq, batch_size=2, seq_len=3)
# inputs = get_inputs(data)

hidden_size = 256
std = 0.01
nd.one_hot(nd.array([0, 2]), vocab_size)

def get_params():
	w_xh = nd.random_normal(scale=std, shape=(vocab_size, hidden_size), ctx=ctx)
	w_hh = nd.random_normal(scale=std, shape=(hidden_size, hidden_size), ctx=ctx)
	b_h = nd.zeros(hidden_size, ctx=ctx)

	w_hy = nd.random_normal(scale=std, shape=(hidden_size, vocab_size), ctx=ctx)
	b_y = nd.zeros(vocab_size, ctx=ctx)

	params = [w_xh, w_hh, b_h, w_hy, b_y]
	for p in params:
		p.attach_grad()
	return params

def rnn(inputs, h, w_xh, w_hh, b_h, w_hy, b_y):
	output = []
	for x in inputs:
		h = nd.tanh(nd.dot(x, w_xh) + nd.dot(h, w_hh) + b_h)
		y = nd.dot(h, w_hy) + b_y
		output.append(y)
	return (output, h)

state = nd.zeros(shape=(data.shape[0], hidden_size), ctx=ctx)

params = get_params()
outputs, state_new = rnn(get_inputs(data.as_in_context(ctx)), state, *params)

print('output length: ',len(outputs))
print('output[0] shape: ', outputs[0].shape)
print('state shape: ', state_new.shape)

