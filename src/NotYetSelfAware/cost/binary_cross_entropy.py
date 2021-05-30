import numpy as np

class BinaryCrossEntropy():
	def __init__(self) -> None:
		pass

	def cost(self, A, y):
		m = y.shape[1]

		# print(f"A: {A.shape}")
		# print(f"y: {y.shape}")
		log_true = np.dot(np.log(A), y.T)
		log_false = np.dot(np.log(1 - A), 1 - y.T)
		log_prob = log_true + log_false

		J = - (1 / m) * log_prob
		J = float(np.squeeze(J))
		
		return J
