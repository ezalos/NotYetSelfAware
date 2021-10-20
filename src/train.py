from NotYetSelfAware.datasets import Datasets
from NotYetSelfAware.preprocessing import Standardize
from NotYetSelfAware.model import Model
from NotYetSelfAware.layers import Dense, Output
from NotYetSelfAware.cost import CrossEntropy
from NotYetSelfAware.config import config
from NotYetSelfAware.optimizers import StochasticGradientDescent, Adam
import pickle

import argparse

def load_dataset(csv_path):
	print(f"Loading: {csv_path}")
	X, y = Datasets().mlp(csv_path=csv_path, y_matrix=True)
	std_X = Standardize()
	std_X.fit(X)
	X = std_X.apply(X)
	print(f"X: {X.shape} dtype -> {X.dtype}")
	print(f"y: {y.shape} dtype -> {y.dtype}")
	return X, y


def build_model(model_args, data_dim, model_config):
	model = Model(**model_args)

	act = model_config['activation']
	units = model_config['units']
	for i in range(model_config['layers']):
		if i == 0:
			model.add_layer(Dense(units, input_dim=data_dim[0], activation=act))
		elif i == model_config['layers'] - 1:
			model.add_layer(Output(data_dim[1], activation="Softmax"))
		else:
			model.add_layer(Dense(units, activation=act))

	loss = CrossEntropy()
	opt = Adam()
	model.compile(optimizer=opt, loss=loss)

	return model


def train(model_args):
	epochs = 2 ** 14

	data_dim = (X.shape[0], y.shape[0])


	model = build_model(args, model_args, data_dim)

	model.score(X, y)
	model.fit(X=X, y=y, epoch=epochs, minibatch=None, train_test_split=0.95)


if __name__ == "__main__":
	# Parsing main arguments
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-d", "--dataset", help="path to .csv", type=str, default=config.mlp_dataset_path)
	parser.add_argument(
		"-lr", "--learning_rate", help="Model learning rate", type=float, default=5e-4)
	parser.add_argument(
		"-v", "--visu", help="Visualize training of Neural Net", action="store_true", default=False)
	parser.add_argument(
		"-s", "--save", help="Model save path", default="")
	args = parser.parse_args()

	# X, y = load_dataset(args.dataset)
	X, y = Datasets().mlp(csv_path=args.dataset, y_matrix=True)
	std_X = Standardize()
	std_X.fit(X)
	X = std_X.apply(X)

	in_out_dims = (X.shape[0], y.shape[0])

	model_args = {
		'learning_rate': args.learning_rate,
		'seed': 42,
		'early_stopping': True,
		'lr_decay': False,
		'weight_decay': 0,  # .15,
		'scores': ["accuracy"],
		# 'save_file': "default",
		'visu': args.visu,
	}
	model_config = {
		'layers' : 5,
		'units' : 32,
		'activation': "tanh",
	}

	model = build_model(model_args, in_out_dims, model_config)

	epochs = 2 ** 14
	model.score(X, y)
	model.fit(X=X, y=y, epoch=epochs, minibatch=None, train_test_split=0.95)

	cache = {
		# 'model': model.best['model'] if model.best['model'] else model.layers,
		'preprocessing': std_X,
		'hyperparams': [model_args, model_config],
	}

	if args.save:
		path = args.save
	else:
		dir = "src/cache/"
		path = f"{dir}model_cache.pkl"

	with open(path, 'wb') as f:
		pickle.dump(cache, f)

	model.save(f"{dir}model.pkl")

