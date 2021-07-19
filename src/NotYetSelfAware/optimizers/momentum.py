from .base import BaseOptimizer

class Momentum(BaseOptimizer):
	def __init__(self, beta=0.9) -> None:
		self.beta = beta
		self.moments = None

	def init_moments(self, layers):
		# Exponentially weighted averages
		self.moments = []
		for l in layers:
			elems = {}
			for param in l.params.keys():
				elems['Vd' + param] = l.grads['d' + param]
			self.moments.append(elems)

	def update(self, layers:list, learning_rate):
		if self.moments == None:
			self.init_moments(layers)
		b = self.beta
		for m, l in zip(self.moments, layers):
			for param in l.params.keys():
				m['Vd' + param] = b * m['Vd' + param] + (1 - b) * l.grads['d' + param]
				l.params[param] = l.params[param] - (learning_rate * m['Vd' + param])
