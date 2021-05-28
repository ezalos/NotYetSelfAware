import numpy as np
from .base import BaseActivation


class Sigmoid(BaseActivation):
	def __init__(self) -> None:
		pass

	def forward(self, Z):
		A = 1 / (1 + np.exp(-Z))
		return A

	def backward(self, Z):
		A = self.forward(A)
		dA = A * (1 - A)
		return dA
