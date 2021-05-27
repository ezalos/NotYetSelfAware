import numpy as np

class Sigmoid():
	def __init__(self) -> None:
		pass

	def forward(self, Z):
		A = 1 / (1 + np.exp(-Z))
		return A

	def backward(self, X):
		Z = self.forward(X)
		dZ = Z * (1 - Z)
		return dZ
