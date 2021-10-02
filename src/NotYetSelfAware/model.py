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
from typing import List
from config import config
import pickle
import copy
import os
from datasets import get_dummies
import matplotlib.pyplot as plt

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
	def __init__(self, 
				 learning_rate: float,
				 seed: int = None,
				 early_stopping:bool = False,
				 lr_decay = False,#str = "",
				 weight_decay: float = 0,
				 scores: List[str] = ["accuracy"],
				#  save_file: str = "default",
				 visu: bool = False) -> None:
		self.seed = seed		
		self.layers =  []

		self.lr_0 = learning_rate
		self.lr = self.lr_0
		self.loss = None
		self.optimizer = None

		self.early_stopping = early_stopping
		self.lr_decay = lr_decay
		self.weight_decay = weight_decay

		self.history = {'loss': []}
		self.verbose_update = {}
		self.scores = scores

		for s in scores:
			self.history[s] = []

		before_val = list(self.history.keys())
		for k in before_val:
			self.history['val_' + k] = []

		# self.save_file = save_file
		
		self.threshold = None
		self.visu_on = visu

		self.best = None

		self._is_compiled = False
		if self.visu_on:
			self.visu = NeuralNetworkVisu()

	def add_layer(self, layer):
		if len(self.layers) == 0 and layer.input_dim == None:
			raise ValueError("Your first layer must have input_dim set")
		self.layers.append(layer)
		logging.debug(f"Added a new layer!\n\t{self.layers[-1]}")

	def _compile_layers(self):
		self.layers[0]._init_build()
		for i in range(1, len(self.layers)):
			prev_layer = self.layers[i - 1]
			curr_layer = self.layers[i]
			if curr_layer.input_dim == None:
				curr_layer.input_dim = prev_layer.n_units
			elif curr_layer.input_dim != prev_layer.n_units:
				msg = "Adjacent layers must be of shapes l1: (b, a) and l2: (c, b)\n"
				msg += f"\tLayer n*{i - 1} is ({prev_layer.n_units},{prev_layer.input_dim})\n"
				msg += f"\tLayer n*{i} is ({curr_layer.n_units},{curr_layer.input_dim})\n"
				raise ValueError(msg)
			curr_layer._init_build()

	def _compile_visu(self):
		self.visu.add_layer(self.layers[0].shape[1])
		for layer in self.layers:
			self.visu.add_layer(layer.shape[0])

	def compile(self,
				loss=BinaryCrossEntropy(),
				optimizer=BaseOptimizer()):
		if self.seed:
			np.random.seed(seed=self.seed)
		self._compile_layers()
		self.loss = loss
		self.optimizer = optimizer
		if self.visu_on:
			self._compile_visu()
		f_pred = None
		if self.layers[-1].g_name.lower() == "sigmoid":
			self.threshold = 0.5
			def f_pred(A, th=0.5):
				return (A > th).astype(np.int64)
		elif self.layers[-1].g_name.lower() == "softmax":
			def f_pred(A):
				categories = A.argmax(axis=0)
				res = get_dummies(categories, range(self.layers[-1].n_units))
				return res
		self.f_pred = f_pred
		self._is_compiled = True

	def _get_minibatch(self, X, y, size):
		if size == None:
			size = X.shape[1]
		m = X.shape[1] // size
		for i in range(m):
			start = size * i
			end = size * (i + 1)
			X_batch = X[:, start:end]
			y_batch = y[:, start:end]
			yield (X_batch, y_batch)

	def _forward(self, X_batch):
		A = X_batch
		for i, l in enumerate(self.layers):
			logging.debug(f"\tForward: layer n*{i}")
			A = l.forward(A)
			if np.isnan(A).any():
				print(f"Activation of layer {i} has NaNs")
		return A

	def _cost(self, AL, y_batch, test=False):
		loss = self.loss.cost(AL, y_batch)
		logg = "loss"
		if test:
			logg = "val_" + logg
		self.history[logg].append(loss)
		self.verbose_update[logg] = loss

	def _backward(self, AL, X_batch, y_batch):
		simplify = True
		if simplify == True:
			dAL = AL - y_batch
		else:
			dAL = self.loss.backward(AL, y_batch)
		self.layers[-1].backward(dAL, self.layers[-2].cache['A'], simplify=simplify)
		for i in range(len(self.layers[:-1]))[::-1]:
			# logging.debug(f"\tBackward: layer n*{i}")
			if i == 0:
				A_m1 = X_batch
			else:
				A_m1 = self.layers[i - 1].cache['A']
			self.layers[i].backward(self.layers[i + 1].params,
									self.layers[i + 1].grads,
									A_m1)

	def _early_stopping(self, epoch):
		if not self.early_stopping:
			return False
		metric = "val_loss"
		max_trys = 100

		if not self.best or self.best[metric] > self.history[metric][-1]:
			# del self.best
			# self.best = None
			self.best = {
				metric: self.history[metric][-1],
				'epoch': epoch,
				'model': copy.deepcopy(self.layers),
			}

		if self.best['epoch'] + max_trys < epoch:
			print("Early stopping")
			print(f"\tBest {metric} is {self.best[metric]} at epoch {self.best['epoch']}")
			if self.visu_on:
				self.visualize(epoch)
				self.visu.exit()
			return True

		return False

	def _verbose(self, pbar, e, jmp):
		pbar.set_postfix(**self.verbose_update)
		if e % jmp == 0:
			if e >= jmp * 2:
				if jmp < 1000:
					jmp = e
			if self.visu_on:
				self.visualize(e)
		return jmp

	def _train_test_split(self, X, y, test_ratio):
		ratio = int(X.shape[1] * test_ratio)
		X_train = X[:,ratio:]
		X_test =  X[:,:ratio]
		Y_train = y[:,ratio:]
		Y_test =  y[:,:ratio]
		return X_train, X_test, Y_train, Y_test

	# TODO: Create decorators to protect user interfaces
	def fit(self, X, y, epoch=1, minibatch=None, verbose:bool=True, train_test_split: float =0):
		if not self._is_compiled:
			raise Exception(f"Model needs to be compiled before being used")
		jmp = 1
		if train_test_split:
			X_train, X_test, Y_train, Y_test = self._train_test_split(X, y, train_test_split)

		with trange(epoch) as pbar:
			pbar.set_description("Fit <3")
			for e in pbar:
				for X_batch, y_batch in self._get_minibatch(X_train, Y_train, minibatch):

					AL = self._forward(X_batch)
					self._cost(AL, y_batch)
					self._backward(AL, X_batch, y_batch)

					self._update(e)
					self.score(X_train, Y_train)

					if train_test_split:
						self.score(X_test, Y_test, test=True)
						self._cost(self._forward(X_test), Y_test, test=True)

					jmp = self._verbose(pbar, e, jmp)
					if self._early_stopping(e):
						return
		if self.visu_on:
			self.visu.exit()
	
	def score(self, X, y, test=False):
		pred = self.predict(X, Threshold=self.threshold)
		# print(f"{pred.shape = }")
		scores = self.history.keys()
		if "accuracy" in scores:
			acc = accuracy(y, pred)
			logg = "accuracy"
			if test:
				logg = "val_" + logg
			self.history[logg].append(acc)
			self.verbose_update[logg] = acc
		

	def predict(self, X, Threshold=None):
		A = self._forward(X)
		# print(f"{A.shape = }")
		if self.f_pred:
			A = self.f_pred(A)
			# A = (A > Threshold).astype(np.int64)
		y_pred = A
		return y_pred

	def _lr_decay(self, epoch):
		decay_rate = 0.5
		self.lr = (1 / (1 + (decay_rate * epoch))) * self.lr_0

	def _update(self, epoch):
		if self.lr_decay:
			self._lr_decay(epoch)
		self.optimizer.update(self.layers)
		weight_decay = self.weight_decay
		for layer, opti in zip(self.layers, self.optimizer.cache):
			for param in layer.params.keys():
				layer.params[param] = ((1 - weight_decay) * layer.params[param]) - \
					(self.lr * opti['d' + param])
				# layer.params[param] = (layer.params[param]) - (self.lr * opti['d' + param])

	def visualize(self, e):
		self.visu.update_weights(self.layers, 
							self.history['loss'], 
							self.history['val_loss'],
							self.history['accuracy'],
							self.history['val_accuracy'])
		self.visu.draw(e)

	def save(self, path):
		# path = config.cache_folder + self.save_file + '.pkl'
		if not os.path.exists(config.cache_folder):
			print(f'{config.cache_folder} does not exist: creating it')
			os.makedirs(config.cache_folder)

		if self.best:
			self.layers = self.best['model']

		with open(path, 'wb') as f:
			pickle.dump(self, f)#, protocol=pickle.HIGHEST_PROTOCOL)

	def load(self, save_file=""):
		save_file = save_file if save_file else self.save_file
		path = config.cache_folder + save_file + '.pkl'
		with open(path, 'rb') as f:
			self = pickle.load(f)
		return self

	def __str__(self) -> str:
		n = 10
		msg = ""
		msg += f"{'~' * n} MODEL {'~' * n}\n"
		msg += f"Loss: {self.loss.__class__.__name__}\n"
		msg += f"Optimizer: {self.optimizer.__class__.__name__}\n"
		for l in self.layers:
			msg += f"{l}\n"
		msg += f"Learning rate: {self.lr}\n"
		msg += f"seed: {self.seed}\n"
		msg += f"{'~' * n}~~~~~~~{'~' * n}"
		return msg

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

	model = Model(learning_rate=1e-1)
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
