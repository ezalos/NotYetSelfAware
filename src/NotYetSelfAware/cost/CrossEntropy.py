import numpy as np
import logging
from .base import BaseCost

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CrossEntropy(BaseCost):
	def __init__(self) -> None:
		pass

	def cost(self, A, y):
		e = 1e-20
		# e = 0
		m = y.shape[1]

		logger.debug(f"{A.shape = }")
		logger.debug(f"{y.shape = }")

		# eps = np.zeros_like(A)
		# eps[A == 0] = e
		eps = e
		log_true = np.dot(np.log(A + eps), y.T)

		# eps = np.zeros_like(A)
		# eps[A == 1] = 1 - e
		log_false = np.dot(np.log(1 - A + eps), (1 - y).T)

		log_prob = log_true + log_false

		J = - (1 / m) * log_prob
		# J = float(np.squeeze(J))
		# print(f"{J = }")
		J = np.squeeze(J)
		# print(f"{J = }")
		# J = float(J.sum())
		logger.debug(f"{J = }")

		return J

	def backward(self, A, Y):
		# return A - Y
		e = 1e-20
		m = Y.shape[1]
		# print(f"{A.shape = }")
		# print(f"{Y.shape = }")
		# eps = np.zeros_like(A)
		# eps[A == 0] = e
		left = - (Y / (A + e))
		# eps = np.zeros_like(A)
		# eps[A == 1] = e
		right = ((1 - Y) / (1 - A + e))
		res = (1/m) * (left + right)
		# print(f"{res.shape = }")
		# res = res.sum(axis=1, keepdims=True)
		# print(f"{res.sum().shape = }")
		return res
