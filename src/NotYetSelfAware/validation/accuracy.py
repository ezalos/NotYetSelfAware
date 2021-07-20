import numpy as np

def accuracy(Y, predictions):
	y_true = np.dot(Y, predictions.T)
	y_false = np.dot(1 - Y, 1 - predictions.T)
	size = float(Y.shape[1])
	accuracy = float((y_true + y_false) / size)
	return accuracy
