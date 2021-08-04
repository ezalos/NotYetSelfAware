import numpy as np
from .activations import Tanh, Sigmoid, ReLU, LeakyReLU, LU, Softmax
# from .activations.sigmoid import Sigmoid


class BaseLayer():
	name = "Base"
	def __init__(self,
              n_units: int,
              input_dim: int = None,
              activation="LeakyReLU"):
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
		self.n_units = n_units
		self.input_dim = input_dim
		self.g_name = activation
		self.shape = None

	def _init_build(self):
		if self.input_dim == None:
			raise ValueError("input_dim must have been assigned")
		self.shape = (self.n_units, self.input_dim)
		self._init_activation(self.g_name)
		self._init_params(self.n_units, self.input_dim)
		self._init_cache()
		self._init_grads()

	def _init_weights(self):
		if self.g_name == "tanh":
			return (1 / self.shape[1]) ** (1/2)
		# elif self.g_name == "ReLU":
		return (2 / self.shape[1]) ** (1/2)
		# Other method
		# return (2 / (input_dim + n_units)) ** (1/2)


	# * Naming convention:
	#	https://stackoverflow.com/questions/8689964/why-do-some-functions-have-underscores-before-and-after-the-function-name
	def _init_params(self, n_units, input_dim):
		self.params = {
			'W': np.random.randn(n_units, input_dim) * self._init_weights(),
			# 'W': np.zeros((n_units, input_dim)) * self._init_weights(),
			'b': np.zeros((n_units, 1)),
		}

	def _init_cache(self):
		self.cache = {
			'Z': None, #np.zeros((n_units, 1)),
			'A': None, #np.zeros((n_units, 1)),
		}

	def _init_grads(self):
		self.grads = {
			'dW': None,
			'db': None,
			'dZ': None,
			# 'dA': None,
		}

	def _init_activation(self, activation):
		activations = {
			"tanh": Tanh,
			"sigmoid": Sigmoid,
			"relu": ReLU,
			"leakyrelu": LeakyReLU,
			"lu": LU,
			"softmax": Softmax,
			}
		activation = activation.lower()
		if activation not in activations.keys():
			raise ValueError(
				f"Error: function activation '{activation}' is not recognized")
		else:
			self.g = activations[activation]()

	def forward(self, dA_m1):
		raise NotImplemented

	def backward(self, dA_m1, dA_p1, opti=True):
		raise NotImplemented

	def __str__(self) -> str:
		msg = f"{' ' * 4}"
		msg += f"{self.name}: "
		msg += f"{self.shape}"
		msg += f"\tg() -> {self.g_name}"
		return msg

