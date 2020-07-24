import numpy as np
from models import KNN, NN


examples = ['I loved the film', 'I fell asleep halfway through']


def create_data_and_targets(file_path, polarity):
	with open(file_path, 'r') as f:
		X = f.read().splitlines()

	if polarity == 'pos':
		y = np.ones(len(X))
	else:
		y = np.zeros(len(X))

	return X, y


def knn_classifier(X, y):
	"""
	K Nearest Neighbours classifier
	Train and test given the entire data
	Predict classes for the provided examples
	"""
	knn = KNN(X,y)
	knn.train()

	print(knn.evaluate())

	knn.predict_for_examples(examples)

def nn_classifier(X, y):
	"""
	Neural Network classifier
	Train and test given the entire data
	Predict classes for the provided examples
	"""
	nn = NN(X,y)
	nn.tokenize()
	nn.train()
	print(nn.evaluate())

	nn.predict_for_examples(examples)

# DO NOT MODIFY THE MAIN METHOD
if __name__ == '__main__':
	X_neg, y_neg = create_data_and_targets('rt-polarity.neg', 'neg')
	X_pos, y_pos = create_data_and_targets('rt-polarity.pos', 'pos')

	X = X_neg + X_pos
	y = np.concatenate((y_neg, y_pos))

	assert len(X) == len(y)

	knn_classifier(X, y)
	nn_classifier(X, y)
