import numpy as np
from .activations import Tanh, Sigmoid, ReLU, LeakyReLU
# from .activations.sigmoid import Sigmoid


class Base():
	name = "Base"
	def __init__(self,
              n_units: int,
              input_dim: int,
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
		self.n_units = n_units
		self.shape = (n_units, input_dim)
		self.learning_rate = learning_rate
		self._init_activation(activation)
		self._init_params(n_units, input_dim)
		self._init_cache(n_units)
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

	def _init_cache(self, n_units):
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
			"ReLU": ReLU,
			"LeakyReLU": LeakyReLU,
			}
		if activation not in activations.keys():
			raise ValueError(
				f"Error: function activation '{activation}' is not recognized")
		else:
			self.g_name = activation
			self.g = activations[activation]()

	def forward(self, dA_m1):
		raise NotImplemented

	def backward(self, dA_m1, dA_p1, opti=True):
		raise NotImplemented

	def update(self, params, grads):
		params['W'] = params['W'] - (self.learning_rate * grads['dW'])
		params['b'] = params['b'] - (self.learning_rate * grads['db'])

	def __str__(self) -> str:
		msg = f"{' ' * 2}"
		msg += f"{self.name}: layer"
		msg += f" of shape {self.shape}"
		msg += f" with activation {self.g_name}"
		return msg

