import matplotlib.pyplot as plt
import matplotlib.image as img
import sys
import attributes_dataset as dataset
import tensorflow as tf
from models import vgg_vertically_sliced as model
import numpy as np
import os
from sklearn import metrics
import logreg

EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1.0

if __name__ == '__main__':
	X_train = dataset.read_convolutional_features(sys.argv[1], sys.argv[2], 'train')
	y_train = dataset.read_labels(sys.argv[1], 'train')
	y_train_oh = np.array([[1 if y_train[i] == j else 0 for j in range(2)] for i in range(len(y_train))])
	X_validate = dataset.read_convolutional_features(sys.argv[1], sys.argv[2], 'validate')
	y_validate = dataset.read_labels(sys.argv[1], 'validate')
	y_validate_oh = np.array([[1 if y_validate[i] == j else 0 for j in range(2)] for i in range(len(y_validate))])
	X_test = dataset.read_convolutional_features(sys.argv[1], sys.argv[2], 'test')
	y_test = dataset.read_labels(sys.argv[1], 'test')
	y_test_oh = np.array([[1 if y_test[i] == j else 0 for j in range(2)] for i in range(len(y_test))])
	best_acc = 0
	best_lambda = 0
	for lambda_factor in np.linspace(1e-5, 1e-1, num=20):
		model = logreg.TFLogReg(X_train.shape[1], 2, param_delta=LEARNING_RATE, param_lambda=lambda_factor)
		model.train(X_train, y_train_oh, EPOCHS)
		y_validate_pred = model.eval(X_validate)
		acc = metrics.accuracy_score(y_validate, np.argmax(y_validate_pred, axis=1))
		if acc > best_acc:
			best_acc = acc
			best_lambda = lambda_factor
	X_train = np.append(X_train, X_validate, axis=0)
	y_train = np.append(y_train, y_validate, axis=0)
	y_train_oh = np.append(y_train_oh, y_validate_oh, axis=0)
	model = logreg.TFLogReg(X_train.shape[1], 2, param_delta=LEARNING_RATE, param_lambda=best_lambda)
	model.train(X_train, y_train_oh, EPOCHS)
	y_train_pred = model.eval(X_train)
	y_train_pred = np.argmax(y_train_pred, axis=1)
	y_test_pred = model.eval(X_test)
	y_test_pred = np.argmax(y_test_pred, axis=1)
	print("Train data results:")
	print("\taccuracy: ", metrics.accuracy_score(y_train, y_train_pred))
	print("\tprecision: ", metrics.precision_score(y_train, y_train_pred))
	print("\taverage precision: ", metrics.average_precision_score(y_train, y_train_pred))
	print("\trecall: ", metrics.recall_score(y_train, y_train_pred))
	print("Test data results:")
	print("\taccuracy: ", metrics.accuracy_score(y_test, y_test_pred))
	print("\tprecision: ", metrics.precision_score(y_test, y_test_pred))
	print("\taverage precision: ", metrics.average_precision_score(y_test, y_test_pred))
	print("\trecall: ", metrics.recall_score(y_test, y_test_pred))
	X_test_imgs = dataset.read_images(sys.argv[1], 'test')
	if len(sys.argv) > 3:
		misclassified_output_folder = sys.argv[3]
		for index, image in enumerate(X_test_imgs):
			if y_test_pred[index] != y_test[index]:
				img.imsave(os.path.join(misclassified_output_folder, str(y_test[index]) + "_" + str(index)), image)