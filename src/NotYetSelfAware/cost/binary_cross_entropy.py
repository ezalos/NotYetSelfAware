import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BinaryCrossEntropy():
	def __init__(self) -> None:
		pass

	def cost(self, A, y):
		e = 1e-20
		# e = 0
		m = y.shape[1]

		logger.debug(f"{A.shape = }")
		logger.debug(f"{y.shape = }")
		
		eps = np.zeros_like(A)
		eps[A == 0] = e
		log_true = np.dot(np.log(A + eps), y.T)

		eps = np.zeros_like(A)
		eps[A == 1] = 1 - e
		log_false = np.dot(np.log(1 - A), (1 - y).T)
		
		log_prob = log_true + log_false

		J = - (1 / m) * log_prob
		J = float(np.squeeze(J))
		logger.debug(f"{J = }")

		return J
