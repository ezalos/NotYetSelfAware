# import .__init__
from numpy.random.mtrand import random
from layers import activations
from model import Model
from layers import Dense, Output
import pandas as pd
from validation import accuracy
from optimizers import StochasticGradientDescent, Adam
import numpy as np
import logging
import sys
from preprocessing import Standardize
from config import config
import argparse
from datasets import Datasets
from cost import CrossEntropy, BinaryCrossEntropy

# Parsing main arguments
parser = argparse.ArgumentParser()
parser.add_argument(
	"-c", "--classes", help="Y is matrix shaped", action="store_true", default=True)
parser.add_argument(
	"-lr", "--learning_rate", help="Model learning ratre", type=float, default=5e-4)
parser.add_argument(
	"-d", "--dataset", help="blobs, mlp, digits, iris or a path ending with .csv", type=str, default="mlp")
parser.add_argument(
	"-v", "--visu", help="blobs or mlp", action="store_true", default=False)
parser.add_argument(
	"-s", "--save", help="Model save path", default="")
parser.add_argument(
	"-l", "--load", help="Model load path", default="")

# parser.add_argument(
# 	"-d", "--directory", help="Datasets directory")
args = parser.parse_args()


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def load_dataset(args):
	X, y = Datasets().generate(569, dataset=args.dataset,
							n_features=30, y_matrix=args.classes)
	std_X = Standardize()
	std_X.fit(X)
	X = std_X.apply(X)
	print(f"X: {X.shape} dtype -> {X.dtype}")
	print(f"y: {y.shape} dtype -> {y.dtype}")
	# print(f"{y.shape = }")
	# print(f"{y = }")
	return X, y

def build_model(args, model_args, data_dim):
	model = Model(**model_args)
	act = "tanh"
	act = "leakyrelu"
	act = "relu"

	model.add_layer(Dense(32, input_dim=data_dim[0], activation=act))
	model.add_layer(Dense(8, activation=act))
	# model.add_layer(Dense(8, activation=act))
	# model.add_layer(Dense(3, activation=act))
	if args.classes:
		model.add_layer(Output(data_dim[1], activation="Softmax"))
		loss = CrossEntropy()

	else:
		model.add_layer(Output(data_dim[1], activation="sigmoid"))
		loss = BinaryCrossEntropy()
	opt = Adam()
	# opt = StochasticGradientDescent()
	model.compile(optimizer=opt, loss=loss)
	print(model)
	return model

X, y = load_dataset(args)

# Iris 99 broke

model_args = {
		'learning_rate': args.learning_rate,
        'seed': 42,
        'early_stopping': True,
        'lr_decay': False,
        'weight_decay': 0,#.15,
        'scores': ["accuracy"],
        # 'save_file': "default",
        'visu': args.visu
		}
epochs = 2 ** 14

data_dim = (X.shape[0], y.shape[0])

if args.load:
	model = Model(42).load(args.load)
else:
	model = build_model(args, model_args, data_dim)

model.score(X, y)
model.fit(X=X, y=y, epoch=epochs, minibatch=None, train_test_split=0.95)

if args.save:
	pass
	# model.save(args.save)
