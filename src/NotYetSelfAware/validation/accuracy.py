import numpy as np

def accuracy(Y, predictions):
	return (Y == predictions).mean()
	y_true = np.dot(Y, predictions.T)
	y_false = np.dot(1 - Y, 1 - predictions.T)
	size = float(Y.shape[1])
	accuracy = (y_true + y_false) / size
	# accuracy = float(accuracy.mean())
	accuracy = float(accuracy)
	return accuracy
