import numpy as np
from layers import Dense, Output
from datasets.datasets import Datasets
from validation import accuracy
from cost import BinaryCrossEntropy
from preprocessing import Standardize
from optimizers import BaseOptimizer, Adam
from visualize import NeuralNetworkVisu
import logging
from tqdm import trange
import sys

# LOGGER INITIALISATION
root = logging.getLogger()
logging.basicConfig(filename="mylog.log")
logging.root.setLevel(logging.INFO)

#SETTING LOGGER TO STDOUT
if True:
	handler = logging.StreamHandler(sys.stdout)
	handler.setLevel(logging.INFO)
	# formatter = logging.Formatter(
	# 	'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# handler.setFormatter(formatter)
	root.addHandler(handler)

class Model():
	def __init__(self, lr, seed=None) -> None:
		self.seed = seed
		if seed:
			np.random.seed(seed=seed)
		
		self.layers =  []
		self.f_loss = BinaryCrossEntropy()
		self.lr_0 = lr
		self.lr = self.lr_0
		# self.std_X = Standardize()
		# self.std_y = Standardize()
		self.optimizer = BaseOptimizer()
		self.losses = []
		self.accues = []
		self.best_acc = -1
		self.best_acc_ep = -1
		self.threshold = 0.5

		self.visu_on = True
		if self.visu_on:
			self.visu = NeuralNetworkVisu()

	def add_layer(self, layer):
		self.layers.append(layer)
		logging.debug(f"Added a new layer!\n\t{self.layers[-1]}")
		if self.visu_on:
			if len(self.visu.layers) == 0:
				self.visu.add_layer(layer.shape[1])
			self.visu.add_layer(layer.shape[0])

	def visualize(self, e):
		self.visu.update_weights(self.layers, self.losses, self.accues)
		self.visu.draw(e)
			

	def get_minibatch(self, X, y, size):
		# self.std_X.fit(X)
		# self.std_y.fit(y)
		if size == None:
			size = X.shape[1]
		# print(X.shape)
		m = X.shape[1] // size
		# print(m)
		for i in range(m):
			start = size * i
			end = size * (i + 1)
			X_batch = X[:, start:end]
			y_batch = y[:, start:end]
			# X_batch = self.std_X.apply(X_batch)
			# y_batch = self.std_y.apply(y_batch)
			# sys.exit()
			yield (X_batch, y_batch)

	def forward(self, X_batch):
		A = X_batch
		for i, l in enumerate(self.layers):
			logging.debug(f"\tForward: layer n*{i}")
			A = l.forward(A)
		return A

	def cost(self, AL, y_batch):
		loss = self.f_loss.cost(AL, y_batch)

		self.losses.append(loss)

	def backward(self, AL, X_batch, y_batch):
		dAL = self.f_loss.backward(AL, y_batch)
		self.layers[-1].backward(dAL, self.layers[-2].cache['A'])
		for i in range(len(self.layers[:-1]))[::-1]:
			# logging.debug(f"\tBackward: layer n*{i}")
			if i == 0:
				A_m1 = X_batch
			else:
				A_m1 = self.layers[i - 1].cache['A']
			self.layers[i].backward(self.layers[i + 1].params,
									self.layers[i + 1].grads,
									A_m1)

	def save(self):
		pass

	def early_stopping(self, epoch):
		# TODO: cleaner solution
		if self.best_acc < self.accues[-1]:
			self.best_acc = self.accues[-1]
			self.best_acc_ep = epoch
			# if self.visu_on:
			# 	self.visualize()
		# else:
			# if self.best_acc_ep + 100 < epoch:
				# print("Early stopping")
				# print(f"\tBest acc is {self.best_acc} at epoch {self.best_acc_ep}")
				# self.visualize()
				# self.visu.exit()
				# return True
		return False

	def fit(self, X, y, epoch=1, minibatch=None):
		self.losses = []
		self.accues = []
		self.epochs = []
		jmp = 1
		# self.visualize(0)
		with trange(epoch, unit="Epochs") as pbar:
			for e in pbar:
				pbar.set_description(f"Epoch {e}")
				# root.info(f"Epoch {e + 1}/ {epoch}")

				for i, (X_batch, y_batch) in enumerate(self.get_minibatch(X, y, minibatch)):
					AL = self.forward(X_batch)

					self.cost(AL, y_batch)
					# loss = self.f_loss.cost(AL, y_batch)
					# if True:
					# 	loss += 

					self.backward(AL, X_batch, y_batch)

					self._update(e)

					self.score(X, y)
					# self.losses.append(loss)
					self.epochs.append(e)

					if self.early_stopping(e):
						return

				pbar.set_postfix(loss=self.losses[-1], accuracy=self.accues[-1])
				if e % jmp == 0:
					if e >= jmp * 2:
						jmp = e
						# print(f"{self.lr = }")
					if self.visu_on:
						self.visualize(e)
				# print(f"Updated!")
		if self.visu_on:
			self.visu.exit()
	
	def score(self, X, y):
		pred = self.predict(X, Threshold=self.threshold)
		acc = accuracy(y, pred)
		# if acc > 0.63:
		# 	self.threshold = self.test_threshold(X, y)
			# for i, (y_, p) in enumerate(zip(y[0], pred[0])):
			# 	if y_ != p:
			# 		print(f"{i}\ty:{y_} {p}:p")
		self.accues.append(acc)

	def test_threshold(self, X, y):
		sav = []
		for t in [0.01 * i for i in range(1, 100)]:
			pred = self.predict(X, Threshold=t)
			acc = accuracy(y, pred)
			sav.append([t, acc])
		sav.sort(key=lambda x:x[1])
		# print(f"Acc = {sav[0][1]} for t = {sav[0][0]}")
		# print(f"Acc = {sav[-1][1]} for t = {sav[-1][0]}")
		return sav[-1][1]



	def _lr_decay(self, epoch):
		decay_rate = 0.5
		self.lr = (1 / (1 + (decay_rate * epoch))) * self.lr_0
		# print(f"{self.lr} =  (1 / ( 1 + ({decay_rate} * {epoch}) * {self.lr_0})")

	def _update(self, epoch):
		# self._lr_decay(epoch)
		self.optimizer.update(self.layers, self.lr)
		weight_decay = 0 * self.lr
		for layer, opti in zip(self.layers, self.optimizer.cache):
			for param in layer.params.keys():
				layer.params[param] = ((1 - weight_decay) * layer.params[param]) - \
					(self.lr * opti['d' + param])
				# print(f"{l.grads['dW'].sum()}")
			# l.params['W'] = l.params['W'] - (self.lr * opt['dW'])
			# l.params['b'] = l.params['b'] - (self.lr * opt['db'])


	def predict(self, X, Threshold=None):
		# X = self.std_X.apply(X)
		A = self.forward(X)
		if Threshold:
			A = (A > Threshold).astype(np.int64)
		y_pred = A
		# y_pred = self.std_y.unapply(y_pred)
		return y_pred


if __name__ == "__main__":
	m = 569
	n_x = 31
	n_L = 2

	X, y = Datasets().generate(m, dataset="blobs", n_features=n_x, n_targets=n_L)
	# y = np.stack([y, ~y], axis=0).reshape((n_L, m))
	X = X.T
	std = Standardize()
	std.fit(X)
	X = std.apply(X)
	print(f"{X.shape = }")
	print(f"{X.dtype = }")
	print(f"{y.shape = }")
	print(f"{y.dtype = }")

	model = Model(lr=1e-1)
	model.add_layer(Dense(5, n_x))
	# model.add_layer(Dense(10, 20))
	model.add_layer(Dense(3, 5))
	# model.add_layer(Dense(3, 4))
	model.add_layer(Output(1, 3, activation="sigmoid"))

	total_epoch = 0
	acc = 0
	epochs = 10_000
	y_pred = model.predict(X=X, Threshold=0.5)

	acc = accuracy(y, y_pred)
	acc_0 = acc
	print(f'Accuracy: {acc}%')
	while acc < 98:
		model.fit(X=X, y=y, epoch=epochs, minibatch=None)

		y_pred = model.predict(X=X, Threshold=0.5)

		acc = accuracy(y, y_pred)
		print(f'{acc   = }%')
		total_epoch += epochs
		acc = 99
	print(f"Total epochs: {total_epoch}")
	print(f"{acc_0 = }%")
