import numpy as np
from .base import BaseLayer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BatchNormalization(BaseLayer):
	name = "Dense"

	def forward(self, A_m1):
		return self.cache['A']

	def backward(self, params_p1, grads_p1, A_m1):
		# * Batch Norm Derivation:
		# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
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
