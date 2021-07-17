# import .__init__
from model import Model
from layers import Dense, Output
import pandas as pd
import numpy as np
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

print(f"Wassup world, it's the __main__ !")
print(f"\tName: {__name__}")
print(f"\tFile: {__file__}")

dataset_path = "src/NotYetSelfAware/datasets/cache/data.csv"
col_names = [str(i) for i in range(32)]
df = pd.read_csv(dataset_path, names=col_names)
print(df.shape)
df_y = df['1']
print(df_y.shape)
y = df_y 
df_X = df.drop(labels='1')
print(df_X.shape)
sys.exit()

model = Model(lr=1e-3)
model.add_layer(Dense(5, n_x))
model.add_layer(Dense(4, 5))
model.add_layer(Dense(3, 4))
model.add_layer(Output(1, 3, activation="sigmoid"))

total_epoch = 0
acc = 0
epochs = 100
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
	acc = 99
print(f"Total epochs: {total_epoch}")
print(f"{acc_0 = }%")
