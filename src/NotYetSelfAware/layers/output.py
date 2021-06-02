import numpy as np
# from base import Base
from .dense import Dense


class Output(Dense):
	name = "Output"

	def backward(self, A_m1, A, Y, opti=True, last=False):
		if not opti:
			self.Z = self.forward(A_m1)

		m = A.shape[1]

		self.dZ = A - Y
		self.dA = np.dot(self.W.T, self.dZ)

		self.dW = (1 / m) * np.dot(self.dZ, A_m1.T)
		self.db = (1 / m) * np.sum(self.dZ, axis=1, keepdims=True)

		if self.debug:
			print("Backward()")
			print(f"m  = {m}")
			print(f"dZ = {self.dZ}")
			print(f"dA = {self.dA}")
			print(f"dW = {self.dW}")
			print(f"db = {self.db}")
			print()

		return self.dA


if __name__ == "__main__":
	np.random.seed(seed=42)

	prev = 2
	curr = 4
	nb_examples = 10
	X = np.random.randn(prev, nb_examples)

	print(f"Prev layers size: {prev}")
	print(f"Curr layers size: {curr}")
	print(f"X values:\n{X}")
	print()

	layer = Dense(curr, prev, debug=True)
	print()
	print(layer)
	print()
	Y = layer.forward(X)
	print(f"Y= {Y}")
	print()
	dA = layer.backward(X, Y)
	print(f"dA= {dA}")
