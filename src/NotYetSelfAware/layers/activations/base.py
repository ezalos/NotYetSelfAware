class BaseActivation():
	def __init__(self):
		raise NotImplemented

	def forward(self, Z):
		raise NotImplemented

	def backward(self, Z):
		raise NotImplemented
