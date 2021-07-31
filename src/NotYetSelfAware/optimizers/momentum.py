from .base import BaseOptimizer
import numpy as np

class Momentum(BaseOptimizer):
	def __init__(self, beta=0.9, bias_correction=False) -> None:
		self.beta = beta
		self.cache = None
		self.t = 0
		self.bias_correction = bias_correction

	def init_cache(self, layers):
		# Exponentially weighted averages
		self.t = 0
		self.cache = []
		for l in layers:
			elems = {}
			for param in l.params.keys():
				elems['Vd' + param] = np.zeros_like(l.grads['d' + param])
			self.cache.append(elems)

	def update(self, layers:list):
		if self.cache == None:
			self.init_cache(layers)
		b = self.beta
		t = self.t
		for m, l in zip(self.cache, layers):
			for param in l.params.keys():
				m['Vd' + param] = b * m['Vd' + param] + (1 - b) * l.grads['d' + param]
				# Correcting Exponentially weighted average
				if self.bias_correction:
					v_update = m['Vd' + param] / (1 - np.power(b, t))
				else:
					v_update = m['Vd' + param]
				m['d' + param] = v_update
				# l.params[param] = l.params[param] - (learning_rate * v_update)
		self.t += 1
