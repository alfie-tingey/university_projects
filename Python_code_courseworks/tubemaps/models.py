import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import plot_model



class Classifier:

	def __init__(self, X, y):
		self.X = X
		self.y = y
		self.model = None


	def split_train_test(self, test_size):
		"""
		split the data into training/testing data and labels

		:param test-size: the percentage of the data that should be used for testing
		"""
		import sklearn
		from sklearn.model_selection import train_test_split

		assert len(self.X) == len(self.y)

		X_train, X_test, y_train, y_test = train_test_split(
			self.X,self.y,
			test_size=test_size,
			random_state = 12,stratify = self.y)

		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test

	def train(self):
		raise NotImplementedError


	def evaluate(self):
		raise NotImplementedError


	# DO NOT MODIFY THIS METHOD
	def predict_for_examples(self, examples):
		return self.model.predict(examples)



class KNN(Classifier):

	def __init__(self, X, y):
		super().__init__(X, y)


	def train(self):
		"""
		Performs a 10 fold grid search using a pipeline with KNN
		Trains on the training data
		"""

		self.split_train_test(test_size = 0.1)

		# Main thing in the Pipeline is that I use the cosine metric. It gave me the best accuracy.

		self.movie_knn = Pipeline([
		('tfidf', TfidfVectorizer()),
		('normalizer',StandardScaler(with_mean=False)),
		('knn', KNeighborsClassifier(metric='cosine')),
		])

		k_range = [7,9,11,13,15,17,19,21,23,25,27,29]
		weight_options = ['uniform','distance']
		param_grid = {'knn__n_neighbors':k_range, 'knn__weights':weight_options}
		self.movie_grid = GridSearchCV(self.movie_knn,param_grid,verbose = 0, cv=10, n_jobs = -1)
		#print(self.movie_grid)

		self.movie_grid.fit(self.X_train,self.y_train)

		#print(self.movie_grid.best_score_)
		#print(self.movie_grid.best_params_)

	def evaluate(self):
		"""
		Computes the predictions for the testing data

		:return: classification report
		"""

		self.test_predictions = self.movie_grid.best_estimator_.predict(self.X_test)
		#print(self.test_predictions)

		#print(f'confusion matrix: {confusion_matrix(self.y_test,self.test_predictions)}')
		#print (f'accuracy: {accuracy_score(self.y_test,self.test_predictions)}')
		return classification_report(self.y_test,self.test_predictions)

	def predict_for_examples(self, examples):

		self.examples_predict = self.movie_grid.best_estimator_.predict(examples)

		print(f'Predicted Class for each example knn: {self.examples_predict}')

		return self.examples_predict

class NN(Classifier):


	VOCAB_SIZE = 10000


	def __init__(self, X, y):
		super().__init__(X, y)


	def tokenize(self):
		"""
		Tokenizes data using only VOCAB_SIZE words based on the frequency
		"""

		self.split_train_test(test_size = 0.2)

		self.train_doc = self.X_train

		self.t = Tokenizer(num_words = self.VOCAB_SIZE)
		self.t.fit_on_texts(self.train_doc)

		self.train_encoded = self.t.texts_to_matrix(self.train_doc,mode = 'freq')
		self.test_encoded = self.t.texts_to_matrix(self.X_test,mode = 'freq')

	def train(self):
		"""
		Builds a NN, prints its summary, and plots it
		Trains on the training data with a validation split
		"""

		self.tokenize()

		inputs = Input(shape = (self.train_encoded.shape[1],))
		x = Dense(64, activation = 'relu')(inputs)
		x = Dense(32, activation = 'relu')(x)
		x = Dense(2,activation = 'softmax')(x)

		self.model = Model(inputs,x)

		from keras.utils import to_categorical
		self.y_train = to_categorical(self.y_train,2)
		self.y_test = to_categorical(self.y_test,2)

		self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =  ['accuracy'])

		print(self.model.summary())
		#plot_model(self.model, to_file='network.png')

		self.model_history = self.model.fit(self.train_encoded, self.y_train, batch_size=128, epochs=5, verbose=0, validation_split=0.2)

	def evaluate(self):
		"""
		Computes the predictions for the testing data

		:return: classification report
		"""

		score = self.model.evaluate(self.test_encoded, self.y_test)
		nn_predict = self.model.predict(self.test_encoded)
		nn_predict_classes = np.around(nn_predict,decimals = 0)

		#print(confusion_matrix(self.y_test, nn_predict_classes))
		#print('Test loss:',score[0])
		#print('Test Accuracy:', score[1])

		return classification_report(self.y_test,nn_predict_classes)

	def predict_for_examples(self, examples):
		"""
		:param examples: a list of strings to classify
		:return: a list of predicted classes for the examples
		"""

		self.tokenize()

		self.examples_encoded = self.t.texts_to_matrix(examples,mode = 'freq')

		examples_predict = self.model.predict(self.examples_encoded)
		examples_predict_classes = np.around(examples_predict,decimals = 0)
		predicted_example_classes = []
		for i in range(len(examples)):
			predicted_example_classes.append(examples_predict_classes[i][1])

		print(f'Predicted Class for each example nn: {predicted_example_classes}')

		return predicted_example_classes
