import numpy as np

class BinaryCrossEntropy():
	def __init__(self) -> None:
		pass

	def cost(self, A, Y):
		m = Y.shape[1]

		log_true = np.dot(np.log(A), Y.T)
		log_false = np.dot(np.log(1 - A), 1 - Y.T)
		log_prob = log_true + log_false

		J = - (1 / m) * log_prob
		J = float(np.squeeze(J))
		
		return J
