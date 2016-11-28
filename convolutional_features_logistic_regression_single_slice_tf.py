import matplotlib.pyplot as plt
import matplotlib.image as img
import sys
import attributes_dataset as dataset
import numpy as np
import os
from sklearn import metrics
import evaluate_helper
from sklearn.linear_model import LogisticRegression

METRIC_FUNCTIONS = (metrics.accuracy_score, metrics.precision_score, metrics.average_precision_score, metrics.recall_score)

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
	best_c = 0
	per_c_metrics_validate = {}
	per_c_metrics_train = {}
	for c_factor in np.logspace(-4, 0, num=50):
		model = LogisticRegression(penalty='l1', C=c_factor, n_jobs=4)
		model.fit(X_train, y_train)
		y_validate_pred = model.predict(X_validate)
		y_train_pred = model.predict(X_train)
		train_metrics = evaluate_helper.evaluate_metric_functions(y_train, y_train_pred, METRIC_FUNCTIONS)
		valid_metrics = evaluate_helper.evaluate_metric_functions(y_validate, y_validate_pred, METRIC_FUNCTIONS)
		per_c_metrics_validate[c_factor] = valid_metrics
		per_c_metrics_train[c_factor] = train_metrics
		print('c=', c_factor)
		print('Train data:')
		print(per_c_metrics_train[c_factor])
		print('Validate data:')
		print(per_c_metrics_validate[c_factor])
		if valid_metrics['accuracy_score'] > best_acc:
			best_acc = valid_metrics['accuracy_score']
			best_c = c_factor
	X_train = np.append(X_train, X_validate, axis=0)
	y_train = np.append(y_train, y_validate, axis=0)
	model = LogisticRegression(penalty='l1', C=best_c, n_jobs=4)
	model.fit(X_train, y_train)
	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)
	train_metrics = evaluate_helper.evaluate_metric_functions(y_train, y_train_pred, METRIC_FUNCTIONS)
	test_metrics = evaluate_helper.evaluate_metric_functions(y_test, y_test_pred, METRIC_FUNCTIONS)
	model_weights = model.coef_
	print("Train lambda table:")
	for c in sorted(per_c_metrics_train.keys()):
		print("c=", c)
		for performance in sorted(per_c_metrics_train[c]):
			print("\t{}={}".format(performance, per_c_metrics_train[c][performance]))
	print("Validate lambda table:")
	for c in sorted(per_c_metrics_validate.keys()):
		print("c=", c)
		for performance in sorted(per_c_metrics_validate[c]):
			print("\t{}={}".format(performance, per_c_metrics_validate[c][performance]))
	print('c=', best_c)
	print("number of weights: ", (model_weights.shape[0] * model_weights.shape[1]), "number of zero weights: ", np.sum(np.abs(model_weights) < 1e-9))
	print("Train data results:")
	print(train_metrics)
	print("Test data results:")
	print(test_metrics)
	if len(sys.argv) > 4:
		X_test_imgs = dataset.read_images(sys.argv[1], 'test')
		misclassified_output_folder = sys.argv[4]
		for index, image in enumerate(X_test_imgs):
			if y_test_pred[index] != y_test[index]:
				img.imsave(os.path.join(misclassified_output_folder, str(y_test[index]) + "_" + str(index)), image)