import numpy as np
from .base import BaseActivation


class LeakyReLU(BaseActivation):
	def __init__(self, coef=0.01, undefined=True) -> None:
		self.coef = coef
		self.undefined = undefined
		pass

	def forward(self, Z):
		A = np.maximum(Z * self.coef, Z)
		return A

	def backward(self, Z):
		A = Z
		A[(Z > 0)] = 1
		A[(Z < 0)] = self.coef
		# Undefined case: arbitrary choice
		A[(Z == 0)] = 1 if self.undefined else self.coef
		return A
