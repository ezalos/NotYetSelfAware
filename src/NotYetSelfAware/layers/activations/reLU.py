import numpy as np
from .base import BaseActivation

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ReLU(BaseActivation):
	def __init__(self, undefined=True) -> None:
		logger.debug(f"{self.__class__.__name__}: Initialization()")
		self.undefined = undefined
		pass

	def forward(self, Z):
		logger.debug(f"{self.__class__.__name__}: forward()")
		A = np.maximum(0., Z)
		# logger.debug(f"\t{A = }")
		return A

	def backward(self, Z):
		A = Z
		A[(Z > 0)] = 1
		A[(Z < 0)] = 0
		# Undefined case: arbitrary choice
		A[(Z == 0)] = 1 if self.undefined else 0
		return A
