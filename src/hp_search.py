from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

from NotYetSelfAware.model import Model
from NotYetSelfAware.preprocessing import Standardize
from NotYetSelfAware.datasets import Datasets
from NotYetSelfAware.config import config
from train import build_model

class PersClassifier(BaseEstimator, ClassifierMixin):
	"""An example of classifier"""
	def __init__(self, learning_rate=0,
				early_stopping=0,
				lr_decay=0,
				weight_decay=0,
				activation=0,
				units=0,
				layers=0,
				minibatch=0,
            	train_test_split=0,
				seed=0,
				):
		"""
		Called when initializing the classifier
		"""
		self.learning_rate = learning_rate
		self.early_stopping = early_stopping
		self.lr_decay = lr_decay
		self.weight_decay = weight_decay
		self.activation = activation
		self.units = units
		self.layers = layers
		self.minibatch = minibatch
		self.train_test_split = train_test_split
		self.seed = seed

		self._fitted = False
		self._model = None
		self._std_X = None


	def fit(self, X, y=None):
		"""
		This should fit classifier. All the "work" should be done here.

		Note: assert is not a good choice here and you should rather
		use try/except blog with exceptions. This is just for short syntax.
		"""
		check_X_y(X, y[:,0])

		X = X.T
		y = y.T

		model_args = {
			'learning_rate': self.learning_rate,
			'seed': self.seed,
			'early_stopping': self.early_stopping,
			'lr_decay': self.lr_decay,
			'weight_decay': self.weight_decay,
			'seed' : self.seed,
		}
		in_out_dims = (X.shape[0], y.shape[0])
		model_config = {
			'layers': self.layers,
			'units': self.units,
			'activation': self.activation,
		}
		self._model = build_model(model_args, in_out_dims, model_config)

		self._std_X = Standardize()
		self._std_X.fit(X)
		X = self._std_X.apply(X)

		self._model.score(X, y)
		self._model.fit(X=X, y=y, epoch=2**12,
					minibatch=self.minibatch,
					train_test_split=self.train_test_split)
		self._fitted = True
		return self

	def _meaning(self, x):
		# returns True/False according to fitted classifier
		# notice underscore on the beginning
		return self._fitted
		return True

	def predict(self, X, y=None):
		X = check_array(X)

		X = self._std_X.apply(X.T)

		Y = self._model.predict(X)
		return Y

	def score(self, X, y=None):
		X = check_array(X)

		X = self._std_X.apply(X.T)

		AL = self._model._forward(X)
		loss = self._model._cost(AL, y.T)
		self._value = 1 / loss
		return (self._value)


if __name__ == "__main__":
	from sklearn.model_selection import GridSearchCV
	from sklearn.model_selection import RandomizedSearchCV
	from sklearn.utils.estimator_checks import check_estimator
	import random
	import math

	random.seed(0)

	p_minibatch = [None]
	p_minibatch.extend([2 ** i for i in range(2, 8)])
	p_train_test_split = [None]
	p_train_test_split = [0.55]
	p_train_test_split.extend(np.linspace(0.6, 1., 10))

	HP_to_test = {
		'learning_rate': [10**(-i) for i in np.linspace(0.5, 9, 20)],
		'early_stopping': [True],
		'lr_decay': [False],
		'weight_decay': [0],
		'activation': ['tanh', 'relu', 'leakyrelu'],
		'units': [2 ** i for i in range(2, 8)],
		'layers': list(range(3, 10)),
		'minibatch': p_minibatch,
		'train_test_split': p_train_test_split,
		'seed': [1, 3, 7, 9, 42, 69, 420],
	}

	tuned_params = HP_to_test
	print(HP_to_test['learning_rate'])
	# gs = RandomizedSearchCV(PersClassifier(), tuned_params, verbose=4)

	X, y = Datasets().mlp(csv_path=config.mlp_dataset_path, y_matrix=True)
	print(f"X: {X.shape} dtype -> {X.dtype}")
	print(f"y: {y.shape} dtype -> {y.dtype}")
	# gs.fit(X=X.T, y=y.T)

	# print(gs.best_params_)  # {'intValue': -10} # and that is what we expect :)



	from evolutionary_search import EvolutionaryAlgorithmSearchCV
	from sklearn.model_selection import StratifiedKFold
	from sklearn.svm import SVC
	# from HyperParamsOpti import PersClassifier
	import sklearn.datasets
	# from config import GridS_conf, net_config
	import numpy as np

	# data = sklearn.datasets.load_digits()
	# X = data["data"]
	# y = data["target"]

	paramgrid = HP_to_test


	cv = EvolutionaryAlgorithmSearchCV(estimator=PersClassifier(),
									params=paramgrid,
									scoring=None,
									cv=3,
									verbose=1,
									population_size=10,
									gene_mutation_prob=0.10,
									gene_crossover_prob=0.5,
									tournament_size=3,
									generations_number=10,
									n_jobs=4)
	ret = cv.fit(X.T, y.T)
	print(ret)
