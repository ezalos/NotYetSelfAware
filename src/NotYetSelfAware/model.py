import numpy as np
from .dense import Dense
from .datasets.datasets import Datasets

class Model():
	def __init__(self) -> None:
		pass

	def add_layer(self, layer):
		pass

	def fit(self, X, Y):
		pass

	def predict(self, X):
		pass


if __name__ == "__main__":

	
	X, y = Datasets().generate(1_000, dataset="blobs", n_features=5, n_targets=2)

	m = Model()
	m.add()
