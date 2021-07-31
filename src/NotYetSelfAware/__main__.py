# import .__init__
from numpy.random.mtrand import random
from model import Model
from layers import Dense, Output
import pandas as pd
from validation import accuracy
from optimizers import StochasticGradientDescent
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
	"-c", "--classes", help="Y is matrix shaped", action="store_true", default=False)
parser.add_argument(
	"-d", "--dataset", help="blobs or mlp", type=str, default="blobs")
parser.add_argument(
	"-v", "--visu", help="blobs or mlp", action="store_true", default=False)
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
	# print(f"{y.dtype = }")
	return X, y

def build_model(args, model_args, data_dim):
	model = Model(**model_args)
	model.add_layer(Dense(5, input_dim=data_dim[0]))
	if args.classes:
		model.add_layer(Output(data_dim[1], activation="Softmax"))
		loss = CrossEntropy()

	else:
		model.add_layer(Output(data_dim[1], activation="sigmoid"))
		loss = BinaryCrossEntropy()
	model.compile(optimizer=StochasticGradientDescent(), loss=loss)
	print(model)
	return model

X, y = load_dataset(args)

model_args = {'learning_rate': 1e-1,
        'seed': None,
        'early_stopping': False,
        'lr_decay': False,
        'weight_decay': 0,
        'scores': ["accuracy"],
        'save_file': "default",
        'visu': args.visu
		}

epochs = 2**14

data_dim = (X.shape[0], y.shape[0])
model = build_model(args, model_args, data_dim)
model.score(X, y)
model.fit(X=X, y=y, epoch=epochs, minibatch=None)
