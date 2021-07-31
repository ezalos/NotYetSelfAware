from .base import BaseOptimizer
import numpy as np


class Adam(BaseOptimizer):
	def __init__(self, beta_1=0.9, beta_2=0.999, epsilon=10e-8) -> None:
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.t = 1
		self.cache = None

	def init_cache(self, layers):
		# Exponentially weighted averages
		self.t = 1
		self.cache = []
		for l in layers:
			elems = {}
			for param in l.params.keys():
				elems['Vd' + param] = np.zeros_like(l.grads['d' + param])
				elems['Sd' + param] = np.zeros_like(l.grads['d' + param])
			self.cache.append(elems)

	def update(self, layers: list):
		if self.cache == None:
			self.init_cache(layers)
		b_1 = self.beta_1
		b_2 = self.beta_2
		eps = self.epsilon
		t = self.t
		for m, l in zip(self.cache, layers):
			for param in l.params.keys():
				# Calcul of Momentum + RMSprop
				m['Vd' + param] = b_1 * m['Vd' + param] + (1 - b_1) * l.grads['d' + param]
				m['Sd' + param] = b_2 * m['Sd' + param] + (1 - b_2) * np.power(l.grads['d' + param], 2)

				# Correcting Exponentially weighted average
				m['Vd' + param + "_corr"] = m['Vd' + param] / (1 - np.power(b_1, t))
				m['Sd' + param + "_corr"] = m['Sd' + param] / (1 - np.power(b_1, t))

				# Updating param
				m['d' + param] = m['Vd' + param + "_corr"] / (np.sqrt(m['Sd' + param + "_corr"]) + eps)
				# l.params[param] = l.params[param] - (learning_rate * v_update)
		self.t += 1
