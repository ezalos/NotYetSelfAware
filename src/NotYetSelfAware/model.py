from layers.output import Output
import numpy as np
from layers import Dense, Output
from datasets.datasets import Datasets
from validation import accuracy
from cost import BinaryCrossEntropy

class Model():
	def __init__(self, lr=None) -> None:
		self.layers =  []
		self.f_loss = BinaryCrossEntropy()
		self.lr = lr

	def change_lr(self, lr_coef):
		for l in self.layers:
			l.learning_rate *= lr_coef

	def add_layer(self, layer):
		if self.lr:
			layer.learning_rate = self.lr
		self.layers.append(layer)
		print(f"Added a new layer!\n\t{self.layers[-1]}")

	def get_batch(self, X, y, size):
		if size == None:
			size = X.shape[1]
		# print(X.shape)
		m = X.shape[0] // size
		# print(m)
		for i in range(m):
			start = size * i
			end = size * (i + 1)
			X_batch = X[start:end, :]
			y_batch = y[:, start:end]
			yield (X_batch, y_batch)

	def fit(self, X, y, epoch=1, batch=128):
		losses = []
		for e in range(epoch):
			print(f"Epoch {e + 1}/ {epoch}")

			for i, (X_batch, y_batch) in enumerate(self.get_batch(X, y, batch)):
				print(f"{' ' * 4}Batch {i + 1}")
				A = X_batch.T
				for i, l in enumerate(self.layers):
					# print(f"\tForward: layer n*{i} -> {l.W.shape}")
					A = l.forward(A)
				AL = A

				loss = self.f_loss.cost(A, y_batch)
				print(f"\tLoss = {loss}")
				losses.append(losses)
				if np.isnan(loss):
					self.change_lr(1/2)
					continue
				if len(losses) > 1 and losses[-1] > losses[-2]:
					self.change_lr(1/2)
				
				if self.layers[-1].name == "Dense":
					dAL = - (np.divide(y_batch, AL) - np.divide(1 - y_batch, (1 - AL) + 1e9))

					# print(f"\tBackward: layer n*{len(self.layers) - 1} -> {self.layers[-1].W.shape}")
					W_p1 = AL
					A_m1 = self.layers[-2].A
					dA = self.layers[-1].backward(y_batch, A_m1, dAL)
				else:
					dA = self.layers[-1].backward(self.layers[-2].A, A, y_batch)
				# print(f"\t\tdA: {dA.shape}")
				for i, l in enumerate(self.layers[:-1][::-1]):
					# print(f"\tBackward: layer n*{len(self.layers[:-1]) - i - 1} -> {l.W.shape}")

					W_p1 = self.layers[i + 1].W
					if i < len(self.layers[:-1]) - 1:
						# print(f"Normal: {i}")
						A_m1 = self.layers[i - 1].A
					else:
						# print(f"First: {i}")
						A_m1 = X_batch.T

					# print(f"\t\tA[-1]: {A_m1.shape}")
					dA = l.backward(W_p1, A_m1, dA)
					# print(f"\t\tdA: {dA.shape}")

				for l in self.layers:
					l.update()
			# print(f"Updated!")

	def predict(self, X, Threshold=None):
		A = X.T
		for i, l in enumerate(self.layers):
			A = l.forward(A)
		if Threshold:
			A = (A > Threshold)
		return A


if __name__ == "__main__":

	m = 1_000
	n_x = 5
	n_L = 2

	X, y = Datasets().generate(m, dataset="blobs", n_features=n_x, n_targets=n_L)
	print(f"X: {X.shape}")
	print(f"y: {y.shape}")

	model = Model(lr=1e-4)
	model.add_layer(Dense(5, n_x))
	model.add_layer(Dense(4, 5))
	model.add_layer(Dense(3, 4))
	model.add_layer(Dense(1, 3, activation="sigmoid"))

	total_epoch = 0
	acc = 0
	epochs = 10
	while acc < 98:
		model.fit(X=X, y=y, epoch=epochs, batch=128)

		y_pred = model.predict(X=X, Threshold=0.5)

		acc = accuracy(y, y_pred)
		print(f'Accuracy: {acc}%')
		total_epoch += epochs
	print(f"Total epochs: {total_epoch}")
