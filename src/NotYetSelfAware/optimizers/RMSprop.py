from .base import BaseOptimizer
import numpy as np


class RMSprop(BaseOptimizer):
	def __init__(self, beta=0.999, epsilon=10e-8, bias_correction=False) -> None:
		self.beta = beta
		self.epsilon = epsilon
		self.cache = None
		self.bias_correction = bias_correction

	def init_cache(self, layers):
		# Exponentially weighted averages
		self.t = 0
		self.cache = []
		for l in layers:
			elems = {}
			for param in l.params.keys():
				elems['Sd' + param] = np.zeros_like(l.grads['d' + param])
			self.cache.append(elems)

	def update(self, layers: list, learning_rate):
		if self.cache == None:
			self.t = 0
			self.init_cache(layers)
		b = self.beta
		eps = self.epsilon
		t = self.t
		for m, l in zip(self.cache, layers):
			for param in l.params.keys():
				# Calcul of Momentum + RMSprop
				m['Sd' + param] = b * m['Sd' + param] + \
					(1 - b) * np.power(l.grads['d' + param], 2)

				if self.bias_correction:
					v_update = m['Sd' + param] / (1 - np.power(b, t))
				else:
					v_update = m['Sd' + param]
			
				v_update = m['d' + param] / (np.sqrt(v_update) + eps)
				# Updating param
				m['d' + param] = v_update
				# l.params[param] = l.params[param] - (learning_rate * v_update)
