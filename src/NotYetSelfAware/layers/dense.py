import numpy as np
import activations
# from .activations.sigmoid import Sigmoid

class Dense():
	def __init__(self,
				 curr_shape: int,
				 prev_shape: int,
				 seed=None,
				 init_coef=0.01,
				 activation="ReLU",
				 debug=False):
		"""[summary]

		Args:
			in_shape (int): [description]
				
				Describe the matrix W shape.
				sizes: (current_layer, previous_layer)

			out_shape (tuple): [description]
			seed ([type], optional): [description]. Defaults to None.
			init_coef (float, optional): [description]. Defaults to 0.01.
			activation (str, optional): [description]. Defaults to "ReLU".
		"""
		self.debug = debug
		if self.debug:
			print("Initialization of Dense")
		self.shape = (curr_shape, prev_shape)
		if seed:
			np.random.seed(seed=seed)
		self.W = np.random.randn(curr_shape, prev_shape)
		self.b = np.zeros((curr_shape, 1))
		self.Z = np.random.randn(curr_shape, 1)
		self.A = np.random.randn(curr_shape, 1)
		self.activation = activation
		self.f_activation = activations.Sigmoid()
		pass

	def forward(self, dA_m1):
		self.Z = np.dot(self.W, dA_m1) + self.b
		self.A = self.f_activation.forward(self.Z)
		return self.A

	def backward(self, dA_m1, dA_p1, opti=True):
		if not opti:
			self.Z = self.forward(dA_m1)

		m = dA_p1.shape[1]

		self.dZ = np.dot(self.W.T, dA_p1)
		self.dA = np.dot(self.dZ, self.f_activation.backward(self.Z).T)

		self.dW = (1 / m) * np.dot(self.dZ.T, dA_m1)
		self.db = (1 / m) * np.sum(self.dZ, axis=1, keepdims=True)

		if self.debug:
			print("Backward()")
			print(f"m  = {m}")
			print(f"dZ = {self.dZ}")
			print(f"dA = {self.dA}")
			print(f"dW = {self.dW}")
			print(f"db = {self.db}")
			print()

		return self.dA

	def __str__(self) -> str:
		msg = f"{' ' * 2}"
		msg += f"{self.__class__}: layer"
		msg += f" of shape {self.shape}"
		msg += f" with activation {self.activation}"
		return msg



if __name__ == "__main__":
	np.random.seed(seed=42)

	prev = 2
	curr = 4
	X = np.random.randn(prev, 1)

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
