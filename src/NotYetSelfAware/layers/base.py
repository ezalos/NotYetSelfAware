import numpy as np
import activations
# from .activations.sigmoid import Sigmoid


class Base():
	name = "Base"

	def __init__(self,
              curr_shape: int,
              prev_shape: int,
              seed=None,
              init_coef=0.01,
              activation="ReLU",
              learning_rate=0.01,
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
		if seed:
			np.random.seed(seed=seed)
		self.shape = (curr_shape, prev_shape)
		self.learning_rate = learning_rate
		self.init_weights(curr_shape, prev_shape)
		self.init_activation(activation)

	def init_weights(self, curr_shape, prev_shape):
		self.W = np.random.randn(curr_shape, prev_shape)
		self.b = np.zeros((curr_shape, 1))
		# Z and A do not have to be assignated here
		# 	but for clarity purpose I did
		self.Z = np.zeros((curr_shape, 1))
		self.A = np.zeros((curr_shape, 1))

	def init_activation(self, activation):
		# Function Activation
		self.activation = activation
		if self.activation == "tanh":
			self.f_activation = activations.Tanh()
		elif self.activation == "sigmoid":
			self.f_activation = activations.Sigmoid()
		elif self.activation == "ReLU":
			self.f_activation = activations.ReLU()
		elif self.activation == "LeakyReLU":
			self.f_activation = activations.LeakyReLU()
		else:
			raise ValueError(
				f"Error: functrion activation {activation} is not recognized")

	def forward(self, dA_m1):
		raise NotImplemented

	def backward(self, dA_m1, dA_p1, opti=True):
		raise NotImplemented

	def update(self, dW, db):
		self.W = self.W - self.learning_rate * dW
		self.b = self.b - self.learning_rate * db

	def __str__(self) -> str:
		msg = f"{' ' * 2}"
		msg += f"{self.name}: layer"
		msg += f" of shape {self.shape}"
		msg += f" with activation {self.activation}"
		return msg
