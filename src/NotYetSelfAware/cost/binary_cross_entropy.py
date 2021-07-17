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
		log_true = np.dot(np.log(A + e), y.T)
		log_false = np.dot(np.log(1 - A + e), (1 - y).T)
		log_prob = log_true + log_false

		J = - (1 / m) * log_prob
		J = float(np.squeeze(J))
		logger.debug(f"{J = }")
		
		return J
