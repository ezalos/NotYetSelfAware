import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Standardize():
	def __init__(self):
		self.mu = 0.
		self.std = 0.

	def fit(self, data):
		logger.debug(f"Fit: {data.shape = }")
		if len(data.shape) >= 2:
			axis = tuple(range(len(data.shape)))[1:]
			# print(f"{axis = }")
			self.mu = np.mean(data, axis=axis)
			self.std = np.std(data, axis=axis)
			self.mu = np.expand_dims(self.mu, axis=axis)
			self.std = np.expand_dims(self.std, axis=axis)
		else:
			axis = 0
			self.mu = np.mean(data, axis=axis)
			self.std = np.std(data, axis=axis)
		logger.debug(f"\t{self.mu.shape = }")
		logger.debug(f"\t{self.std.shape = }")
		return self

	def apply(self, data):
		logger.debug(f"Apply: {data.shape = }")
		mu_0 = (data - self.mu)
		std_1 = mu_0 / self.std
		logger.debug(f"\t{std_1.shape = }")
		return std_1

	def unapply(self, data):
		return (data * self.std) + self.mu


if __name__ == "__main__":
	# Shape (3,4)
	X = [[1, 10, 100, 1000],
		 [-1, -10, -100, -1000],
		 [1, 2, 3, 4]]
	y = [123, 5321, 67, 809]
	img = [[[0, 1], [2, 3], [4, 5]],
		   [[1, 1], [2, 3], [4, 5]],
		   [[2, 1], [2, 3], [4, 5]],
		   [[4, 1], [2, 3], [4, 5]]]

	tests = [y, X, img]

	for t in tests:
		t = np.array(t)
		print(f"{t = }")
		std = Standardize()
		std.fit(t)
		prep_t = std.apply(t)
		print(f"{prep_t = }")
		same_t = std.unapply(prep_t)
		print(f"{same_t = }")
		if same_t.all() == t.all():
			print(f"{'-' * 10} GREAT {'-' * 10}")
		else:
			print(f"{'!' * 10} ERROR {'!' * 10}")
