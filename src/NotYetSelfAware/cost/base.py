class BaseCost():
	def __init__(self) -> None:
		pass

	def cost(self, A, y):
		raise NotImplementedError

	def backward(self, A, Y):
		raise NotImplementedError
