import numpy as np
# from base import Base
from .dense import Dense
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Output(Dense):
	name = "Output"

	def forward(self, A_m1):
		self.cache['Z'] = np.dot(self.params['W'], A_m1) + self.params['b']
		self.cache['A'] = self.g.forward(self.cache['Z'])

		# logging
		logger.debug(f"{self.__class__.__name__}: forward()")
		return self.cache['A']

	def backward(self, A, Y, A_m1):
		self.grads['dZ'] = (A - Y) * self.g.backward(self.cache['Z'])

		m = self.n_units
		self.grads['dW'] = (1 / m) * np.dot(self.grads['dZ'], A_m1.T)
		self.grads['db'] = (1 / m) * np.sum(self.grads['dZ'], axis=1, keepdims=True)

		# logging
		logger.debug(f"{self.__class__.__name__}: backward()")
		logger.debug(f"\tm  = {m}")
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
