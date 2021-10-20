from NotYetSelfAware.model import Model
from NotYetSelfAware.datasets import Datasets
from NotYetSelfAware.config import config
import argparse
import pickle



if __name__ == "__main__":
	# Parsing main arguments
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-d", "--dataset", help="path to .csv", type=str, default=config.mlp_dataset_path)
	parser.add_argument(
		"-l", "--load", help="Load model cache", default="")
	args = parser.parse_args()

	# cache = {
	# 	'model': model.best['model'] if model.best['model'] else model.layers,
	# 	'preprocessing': std_X,
	# 	'hyperparams': [model_args, model_config],
	# }
	dir = "src/cache/"
	path = f"{dir}model_cache.pkl"
	with open(path, 'rb') as f:
		cache = pickle.load(f)
	model = Model(42).load(f"{dir}model.pkl")

	std_X = cache['preprocessing']
	X, y = Datasets().mlp(csv_path=args.dataset, y_matrix=True)
	X = std_X.apply(X)


	model.score(X, y)

	AL = model._forward(X)
	loss = model._cost(AL, y)

	print(f"Binary cross entropy cost = {loss}")
