from sklearn.datasets import make_blobs, make_moons, make_circles, make_regression, load_iris, load_digits
import pandas as pd
import numpy as np
from NotYetSelfAware.config import config


def get_dummies(y, uniques=None):
	if not uniques:
		uniques = np.unique(y)
	# could also be interesting with np.unique()
	new_y = []
	for i in uniques:
		tmp = np.zeros_like(y)
		tmp[y == i] = 1
		tmp = tmp.reshape(-1)
		new_y.append(tmp)
	y = np.stack(new_y, axis=0)
	return y

class Datasets():
	# Reference: https://machinelearningmastery.com/generate-test-datasets-python-scikit-learn/
	def __init__(self, noise=0.15) -> None:
		self.noise = noise
		pass

	def generate(self, n_examples, n_features=None, dataset="blobs", n_targets=None, y_matrix=False):
		if dataset == "mlp":
			return self.mlp(y_matrix=y_matrix)
		elif dataset.endswith(".csv"):
			return self.mlp(csv_path=dataset, y_matrix=y_matrix)
		elif dataset == "blobs":
			if n_targets == None:
				n_targets = 2
			if n_features == None:
				n_features = 4
			print(f"Loading {dataset}: ({n_features}, {n_targets})")
			return self.blobs(n_examples, n_features, n_targets, y_matrix)
		elif dataset == "moons":
			return self.moons(n_examples)
		elif dataset == "circles":
			return self.circles(n_examples)
		elif dataset == "regression":
			if n_targets == None:
				n_targets = 1
			if n_features == None:
				n_features = 4
			return self.regression(n_examples, n_features, n_targets)
		elif dataset == "iris":
			return self.iris(y_matrix)
		elif dataset == "digits":
			return self.digits(y_matrix)
		else:
			raise ValueError(f"Error there is no dataset generator of type {dataset}")

	def blobs(self, n_examples, n_features, n_targets=2, y_matrix=False):
		# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

		X, y = make_blobs(n_samples=n_examples,
                    centers=n_targets,
					n_features=n_features)
		y = y.reshape((1, -1))
		X = X.T
		if y_matrix:
			y = get_dummies(y)
		return X, y

	def iris(self, y_matrix=False):
		# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
		dataset = load_iris()
		X = dataset['data']
		y = dataset['target']
		print(f"{X.shape = }")
		print(f"{y.shape = }")
		y = y.reshape((1, -1))
		print(f"{y = }")
		X = X.T
		if y_matrix:
			y = get_dummies(y)
		else:
			y[y==2] = 0
		return X, y

	def digits(self, y_matrix=False):
		# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
		dataset = load_digits()
		X = dataset['data']
		y = dataset['target']
		print(f"{X.shape = }")
		print(f"{X = }")
		print(f"{y.shape = }")
		y = y.reshape((1, -1))
		print(f"{y = }")
		X = X.T
		if y_matrix:
			y = get_dummies(y)
		if np.isnan(X).any():
			print(f"Sklearn Dataset has NaNs !!!")
			X[np.isnan(X)] = 0
		# if np.isnan(X).any():
		# 	print(f"Sklearn Dataset still has NaNs !!!")
		# 	test = np.isnan(X).astype(np.int64)
		# 	print(np.max(test))
		# 	# print(np.isnan(X))
		return X, y

	def moons(self, n_examples):
		# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
		X, y = make_moons(n_samples=n_examples, noise=self.noise)
		y = y.reshape((1, -1))
		X = X.T
		return X, y

	def circles(self, n_examples):
		# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html
		X, y = make_circles(n_samples=n_examples, noise=self.noise)
		y = y.reshape((1, -1))
		X = X.T
		return X, y

	def regression(self, n_examples, n_features, n_targets=1):
		# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

		X, y = make_regression(n_samples=n_examples, n_features=n_features,
		                       n_targets=n_targets, noise=self.noise)
		y = y.reshape((1, -1))
		X = X.T
		return X, y

	def mlp(self, csv_path=config.mlp_dataset_path, y_matrix=False):
		# path = config.mlp_dataset_path
		col_names = [str(i) for i in range(32)]
		df = pd.read_csv(csv_path, names=col_names)
		df_y = df['1']
		y = df_y
		df_X = df.drop(labels=['0', '1'], axis=1)
		X = df_X.to_numpy()
		y = df_y.to_numpy()
		y[y == 'M'] = 1
		y[y == 'B'] = 0
		y = y.astype(np.int64)
		y = y.reshape(1, -1)
		X = X.reshape(X.shape[::-1])
		if y_matrix:
			y = get_dummies(y)
		return X, y

if __name__ == "__main__":
	from pandas import DataFrame
	from matplotlib import pyplot as plt

	WAIT_TIME = 1
	def little_plot(X, y):
		# scatter plot, dots colored by class value
		df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
		colors = {0: 'red', 1: 'blue', 2: 'green'}
		fig, ax = plt.subplots()
		grouped = df.groupby('label')
		for key, group in grouped:
			group.plot(ax=ax, kind='scatter', x='x',
						y='y', label=key, color=colors[key])
		plt.show(block=False)
		plt.pause(WAIT_TIME)
		plt.close()
	X, y = Datasets().generate(100, dataset="mlp")
	little_plot(X, y)

	X, y = Datasets().generate(100, dataset="blobs", n_targets=3)
	little_plot(X, y)

	X, y = Datasets().generate(100, dataset="moons")
	little_plot(X, y)

	X, y = Datasets().generate(100, dataset="circles")
	little_plot(X, y)

	X, y = Datasets().generate(100, dataset="regression")
	# plot regression dataset
	plt.scatter(X[:, 0], y)
	plt.show(block=False)
	plt.pause(WAIT_TIME)
	plt.close()
