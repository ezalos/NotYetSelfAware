import numpy as np

class L2():
	def __init__(self, lambda_=1) -> None:
		self.lambda_ = lambda_

	def cost(self, layers):
		reg = 0
		for layer in layers:
			reg += np.sum(layer.params['W'])
		return reg

	def grad(self, layer, m):
		w = layer.params['W']
		reg = np.sum(w)
		return reg * (self.lambda_) / (2 * m)

	
