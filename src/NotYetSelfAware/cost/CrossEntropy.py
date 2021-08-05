from matplotlib.pyplot import axis
import numpy as np
import logging
from .base import BaseCost

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CrossEntropy(BaseCost):
	def __init__(self) -> None:
		pass

	def cost(self, A, Y):
		# Y is of shape (n_x, m)
		e = 1e-20
		m = Y.shape[1]

		# print(f"{A = }")
		logA = np.log(A + e)
		# print(f"{logA = }")
		logA_k = np.sum(logA, axis=0)
		# print(f"{logA_k = }")
		Y_k = np.sum(Y, axis=0)
		# print(f"{Y_k = }")
		
		J = -(Y_k * logA_k).sum() / m
		# print(f"{J = }")

		J = np.squeeze(J)
		return J

	def backward(self, A, Y):
		return (Y / A)
