import numpy as np
from .base import BaseActivation


class Softmax(BaseActivation):
	def __init__(self,shift=False) -> None:
		self.shift = shift
		# self.shift = True
		pass

	def forward(self, Z):
		# * Great einsum explanation:
		# 	https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
		if self.shift:
			# * The exponential can quickly give NaNs
			# * To avoid them we can shift the softmax by substracting it's max value
			# * The resulting array will be composed of 0 and 1, loosing precision
			#	https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
			shiftZ = Z - np.max(Z)
			exp_Z = np.exp(shiftZ)
		else:
			exp_Z = np.exp(Z)
		exp_sum = 1 / np.sum(exp_Z, axis=0)
		A = np.einsum("ij,j->ij", exp_Z, exp_sum)
		return A

	def backward(self, Z):
		

		exp_Z = np.exp(Z)
		exp_sum = np.sum(exp_Z, axis=0)
		dA = Z - np.log(exp_sum)
		print(f"{dA.shape = }")
		return dA

if __name__ == "__main__":
	# Shape is (Classes, Examples)
	# 	Here -> (4, 3)
	Z = np.array([[5, 2, -1, 3],
               	  [7, 2, -1, 3],
               	  [6, 2, -1, 3]]).T
	print(f"{Z.shape = }")
	print(f"{Z = }")
	A = Softmax().forward(Z)
	print(f"{A.shape = }")
	print(f"{A = }")
	dA = Softmax().backward(A)
	print(f"{dA.shape = }")
	print(f"{dA = }")

