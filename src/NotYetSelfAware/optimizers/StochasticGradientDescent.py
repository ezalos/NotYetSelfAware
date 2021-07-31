import numpy as np
from .base import BaseOptimizer


class StochasticGradientDescent(BaseOptimizer):
	def __init__(self) -> None:
		self.cache = None
		self.t = 0

	def init_cache(self, layers):
		self.t = 0
		self.cache = []
		for l in layers:
			self.cache.append(l.grads)

	def update(self, layers: list):
		if self.cache == None:
			self.init_cache(layers)
		self.t += 1
