import numpy as np
from .base import BaseActivation


class ReLU(BaseActivation):
	def __init__(self, undefined=True) -> None:
		print("Initialization of ReLU")
		self.undefined = undefined
		pass

	def forward(self, Z):
		A = np.max(0., Z)
		return A

	def backward(self, Z):
		A = Z
		A[(Z > 0)] = 1
		A[(Z < 0)] = 0
		# Undefined case: arbitrary choice
		A[(Z == 0)] = 1 if self.undefined else 0
		return A
