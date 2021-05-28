import numpy as np
from .base import BaseActivation


class Tanh(BaseActivation):
	def __init__(self) -> None:
		pass

	def forward(self, Z):
		# A = np.tanh(Z)
		up = np.exp(Z) - np.exp(-Z)
		dn = np.exp(Z) + np.exp(-Z)
		A = up / dn
		return A

	def backward(self, Z):
		A = self.forward(Z)
		dA = 1 - np.power(A, 2)
		return dA
