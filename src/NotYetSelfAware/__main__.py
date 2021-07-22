# import .__init__
from numpy.random.mtrand import random
from model import Model
from layers import Dense, Output
import pandas as pd
from validation import accuracy
import numpy as np
import logging
import sys
from preprocessing import Standardize

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def load_dataset(path):
	col_names = [str(i) for i in range(32)]
	df = pd.read_csv(dataset_path, names=col_names)
	df_y = df['1']
	y = df_y
	df_X = df.drop(labels='1', axis=1)
	X = df_X.to_numpy()
	y = df_y.to_numpy()

	# print(y)
	y[y == 'M'] = 1
	# print(y)
	y[y == 'B'] = 0
	# print(y)

	y = y.astype(np.int64)
	y = y.reshape(1, -1)
	X = X.reshape(X.shape[::-1])
	std_X = Standardize()
	std_X.fit(X)
	X = std_X.apply(X)
	print(X)
	print(f"{X.shape = }")
	print(f"{y.shape = }")

	return X, y


def build_model(lr, seed, layers):
	model = Model(lr=lr, seed=seed)
	n_l = len(layers) - 1
	for i in range(n_l):
		if i < n_l -1:
			model.add_layer(Dense(layers[i + 1], layers[i]))
		else:
			model.add_layer(Output(layers[i + 1], layers[i], activation="sigmoid"))
	return model


def lr():
	pow = np.random.randint(1, 7)
	return (10 ** -pow) * np.random.random()


def run():
	seed = np.random.randint(1, 1_000_000_000)
	l_r = 1e-1
	print(f"{l_r = } & {seed = }")
	model = build_model(l_r, seed, layers)
	model.score(X, y)
	model.fit(X=X, y=y, epoch=epochs, minibatch=None)
	y_pred = model.predict(X=X, Threshold=0.5)
	acc = accuracy(y, y_pred)
	return l_r, seed, acc

dataset_path = "src/NotYetSelfAware/datasets/cache/data.csv"

X, y = load_dataset(dataset_path)
seed = None
n_x = X.shape[0]
mid = 10
layers = [n_x, 5, 3, 1]
epochs = (2 ** 16) + 1

total_epoch = 0
acc = 0

# get_val = 
histor = []
import json

while True:
	elem = run()
	histor.append(elem)
	histor.sort(key=lambda a: a[-1], reverse=True)
	print(f"Best lr {histor[0][0]} with {histor[0][-1]} acc -> {histor[0]}")
	# with open('hp.json', 'w+') as f:
	# 	json.dump(histor, f, indent=4)
	# for h in histor:
	# 	print(h[-1])
