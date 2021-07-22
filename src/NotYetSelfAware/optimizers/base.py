import numpy as np

class BaseOptimizer():
	def __init__(self) -> None:
		self.cache = None
		self.t = 0

	def init_cache(self, layers):
		self.t = 0
		self.cache = []
		for l in layers:
			elems = {}
			for param in l.params.keys():
				elems['d' + param] = np.zeros_like(l.grads['d' + param])
			self.cache.append(elems)

	def update(self, layers: list, learning_rate):
		if self.cache == None:
			self.init_cache(layers)
		for m, l in zip(self.cache, layers):
			for param in l.params.keys():
				m['d' + param] = l.grads['d' + param]
		self.t += 1
