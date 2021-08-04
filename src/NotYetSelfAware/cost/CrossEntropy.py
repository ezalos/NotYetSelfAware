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

		logA = np.log(A + e)

		logA_k = np.sum(A, axis=1)
		Y_k = np.sum(Y, axis=1)
		
		J = - (Y_k * logA_k).sum() / m

		J = np.squeeze(J)
		return J

	def backward(self, A, Y):
		# return A - Y
		# return res
		pass
