# import .__init__
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

	y[y == 'M'] = 1
	y[y == 'B'] = 0

	y = y.astype(np.int64)
	y = y.reshape(1, -1)
	X = X.reshape(X.shape[::-1])
	std_X = Standardize()
	std_X.fit(X)
	X = std_X.apply(X)

	print(f"{X.shape = }")
	print(f"{y.shape = }")

	return X, y


def build_model(lr, seed, layers):
	model = Model(lr=1e-4, seed=seed)
	n_l = len(layers) - 1
	for i in range(n_l):
		if i < n_l -1:
			model.add_layer(Dense(layers[i + 1], layers[i]))
		else:
			model.add_layer(Output(layers[i + 1], layers[i], activation="sigmoid"))
	return model




dataset_path = "src/NotYetSelfAware/datasets/cache/data.csv"

X, y = load_dataset(dataset_path)
seed = None
n_x = X.shape[0]
mid = 10
layers = [n_x, n_x, n_x, n_x, 1]
epochs = 10_000

total_epoch = 0
acc = 0

# get_val = 
histor = []

# for i in range(100_000):
	# seed = i
model = build_model(1e-3, seed, layers)
# y_pred = model.predict(X=X, Threshold=0.5)
# acc = accuracy(y, y_pred)
# acc_0 = acc
# print(f'Accuracy 0: {acc_0}%')
model.fit(X=X, y=y, epoch=epochs, minibatch=None)
y_pred = model.predict(X=X, Threshold=0.5)
acc = accuracy(y, y_pred)
histor.append([i, acc])
histor.sort(key=lambda a: a[1], reverse=True)
print(f"Best seed {histor[0][0]} with {histor[0][1]}% acc")

	# print(f'{acc   = }%')
	# total_epoch += epochs
# print(f"Total epochs: {total_epoch}")
# print(f"{acc_0 = }%")
