import numpy as np
from .base import BaseActivation
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Sigmoid(BaseActivation):
	def __init__(self) -> None:
		pass

	def forward(self, Z):
		logger.debug(f"{self.__class__.__name__}: forward()")
		A = 1 / (1 + np.exp(-Z))
		return A

	def backward(self, Z):
		A = self.forward(Z)
		dA = A * (1 - A)
		return dA

	def pred(self, A, th=0.5):
		return (A > th).astype(np.int64)
