import numpy as np
from .base import BaseLayer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Dense(BaseLayer):
	name = "Dense"

	def forward(self, A_m1):
		# * Using 'np.dot()' or '@' ?
		#	https://stackoverflow.com/questions/34142485/difference-between-numpy-dot-and-python-3-5-matrix-multiplication
		logger.debug(f"{self.__class__.__name__}: forward()")
		logger.debug(f"\t{self.params['W'].shape = }")
		logger.debug(f"\t{A_m1.shape = }")

		self.cache['Z'] = np.dot(self.params['W'], A_m1) + self.params['b']

		logger.debug(f"\t{self.cache['Z'].shape = }")

		self.cache['A'] = self.g.forward(self.cache['Z'])
		
		logger.debug(f"\t{self.cache['A'].shape = }")

		# logging
		# logger.debug(f"\t{self.cache['A'] = }")
		return self.cache['A']

	def backward(self, params_p1, grads_p1, A_m1):
		logger.debug(f"{self.__class__.__name__}: Backward()")
		logger.debug(f"\t{self.n_units =}")

		logger.debug(f"\t{grads_p1['dZ'].shape = }")
		logger.debug(f"\t{params_p1['W'].T.shape = }")
		logger.debug(f"\t{self.cache['Z'].shape = }")
		logger.debug(f"\t{self.g.backward(self.cache['Z']).shape = }")


		self.grads['dZ'] = np.dot(params_p1['W'].T, grads_p1['dZ']) * self.g.backward(self.cache['Z'])

		m = self.n_units
		self.grads['dW'] = (1 / m) * np.dot(self.grads['dZ'], A_m1.T)
		self.grads['db'] = (1 / m) * np.sum(self.grads['dZ'], axis=1, keepdims=True)


		logger.debug(f"\tdZ: {self.grads['dZ'].shape}")

		return self.grads['dZ']


if __name__ == "__main__":
	np.random.seed(seed=42)

	prev = 2
	curr = 4
	nb_examples = 10
	X = np.random.randn(prev, nb_examples)

	print(f"Prev layers size: {prev}")
	print(f"Curr layers size: {curr}")
	print(f"X values:\n{X}")
	print()

	layer = Dense(curr, prev, debug=True)
	print()
	print(layer)
	print()
	Y = layer.forward(X)
	print(f"Y= {Y}")
	print()
	dA = layer.backward(X, Y)
	print(f"dA= {dA}")
