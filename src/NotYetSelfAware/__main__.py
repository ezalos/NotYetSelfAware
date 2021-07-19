# import .__init__
from model import Model
from layers import Dense, Output
import pandas as pd
from validation import accuracy
import numpy as np
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

print(f"Wassup world, it's the __main__ !")
print(f"\tName: {__name__}")
print(f"\tFile: {__file__}")

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

	print(f"{X.shape = }")
	print(f"{y.shape = }")

	return X, y

dataset_path = "src/NotYetSelfAware/datasets/cache/data.csv"

X, y = load_dataset(dataset_path)

model = Model(lr=1e-4)
model.add_layer(Dense(30, X.shape[0]))
model.add_layer(Dense(30, 30))
model.add_layer(Dense(30, 30))
model.add_layer(Dense(30, 30))
model.add_layer(Dense(5, 30))
model.add_layer(Output(1, 5, activation="sigmoid"))

total_epoch = 0
acc = 0
epochs = 1000
y_pred = model.predict(X=X, Threshold=0.5)

acc = accuracy(y, y_pred)
acc_0 = acc
print(f'Accuracy: {acc}%')
while acc < 98:
	model.fit(X=X, y=y, epoch=epochs, minibatch=128)

	y_pred = model.predict(X=X, Threshold=0.5)

	acc = accuracy(y, y_pred)
	print(f'{acc   = }%')
	total_epoch += epochs
print(f"Total epochs: {total_epoch}")
print(f"{acc_0 = }%")
