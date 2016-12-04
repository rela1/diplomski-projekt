import time
import matplotlib.pyplot as plt
import matplotlib.image as img
import sys
import attributes_dataset as dataset
import numpy as np
import os
from sklearn import metrics
import evaluate_helper
from sklearn.svm import SVC
from sklearn.externals import joblib

METRIC_FUNCTIONS = (metrics.accuracy_score, metrics.precision_score, metrics.average_precision_score, metrics.recall_score)

def print_metrics(metrics):
	for metric in metrics:
		print('\t{}={}'.format(metric, metrics[metric]))

def measure_sparsity(train, test, valid):
	avg_percentage = 0
	avg_percentage += float(np.sum(np.absolute(train) <= 1E-6))
	avg_percentage += float(np.sum(np.absolute(test) <= 1E-6)) 
	avg_percentage += float(np.sum(np.absolute(valid) <= 1E-6))
	example_shape = train.shape[1]
	return avg_percentage / (len(train) * example_shape + len(test) * example_shape + len(valid) * example_shape)

if __name__ == '__main__':
	X_train_left = dataset.read_convolutional_features(sys.argv[1], sys.argv[2], 'train')
	X_train_middle = dataset.read_convolutional_features(sys.argv[1], sys.argv[3], 'train')
	X_train = np.concatenate((X_train_left, X_train_middle), axis=1)
	y_train = dataset.read_labels(sys.argv[1], 'train')
	X_validate_left = dataset.read_convolutional_features(sys.argv[1], sys.argv[2], 'validate')
	X_validate_middle = dataset.read_convolutional_features(sys.argv[1], sys.argv[3], 'validate')
	X_validate = np.concatenate((X_validate_left, X_validate_middle), axis=1)
	y_validate = dataset.read_labels(sys.argv[1], 'validate')
	X_test_left = dataset.read_convolutional_features(sys.argv[1], sys.argv[2], 'test')
	X_test_middle = dataset.read_convolutional_features(sys.argv[1], sys.argv[3], 'test')
	X_test = np.concatenate((X_test_left, X_test_middle), axis=1)
	y_test = dataset.read_labels(sys.argv[1], 'test')
	print('Data sparsity:', measure_sparsity(X_train, X_test, X_validate))
	best_acc = 0
	best_c = 0
	per_c_train_metrics = {}
	per_c_valid_metrics = {}
	c_range = np.logspace(-2, 2, num=100)
	num_iterations = len(c_range)
	current_iteration = 1
	for c_factor in c_range:
		start = time.clock()
		model = SVC(C=c_factor)
		model.fit(X_train, y_train)
		y_validate_pred = model.predict(X_validate)
		y_train_pred = model.predict(X_train)
		train_metrics = evaluate_helper.evaluate_metric_functions(y_train, y_train_pred, METRIC_FUNCTIONS)
		valid_metrics = evaluate_helper.evaluate_metric_functions(y_validate, y_validate_pred, METRIC_FUNCTIONS)
		per_c_train_metrics[c_factor] = train_metrics
		per_c_valid_metrics[c_factor] = valid_metrics
		print('Current valid acc:{}, current c:{}, iteration:{}/{}, time:{}'.format(valid_metrics['accuracy_score'], c_factor, current_iteration, num_iterations, (time.clock() - start)))
		if valid_metrics['accuracy_score'] > best_acc:
			best_acc = valid_metrics['accuracy_score']
			print('New best valid acc:', best_acc)
			best_c = c_factor
		current_iteration += 1
	X_train = np.append(X_train, X_validate, axis=0)
	y_train = np.append(y_train, y_validate, axis=0)
	model = SVC(C=best_c)
	model.fit(X_train, y_train)
	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)
	train_metrics = evaluate_helper.evaluate_metric_functions(y_train, y_train_pred, METRIC_FUNCTIONS)
	test_metrics = evaluate_helper.evaluate_metric_functions(y_test, y_test_pred, METRIC_FUNCTIONS)
	print("Train data results:")
	print('\tc=', best_c)
	print_metrics(train_metrics)
	print("Test data results:")
	print('\tc=', best_c)
	print_metrics(test_metrics)
	joblib.dump(model, sys.argv[4])
	sorted_c = sorted(per_c_valid_metrics.keys())
	plt.subplot('211')
	plt.plot(sorted_c, [per_c_train_metrics[c]['accuracy_score'] for c in sorted_c])
	plt.xlabel('c')
	plt.ylabel('accuracy')
	plt.title('Train accuracy')
	plt.subplot('212')
	plt.plot(sorted_c, [per_c_valid_metrics[c]['accuracy_score'] for c in sorted_c])
	plt.xlabel('c')
	plt.ylabel('accuracy')
	plt.title('Validation accuracy')
	plt.show()
	if len(sys.argv) > 5:
		X_test_imgs = dataset.read_images(sys.argv[1], 'test')
		misclassified_output_folder = sys.argv[5]
		for index, image in enumerate(X_test_imgs):
			if y_test_pred[index] != y_test[index]:
				img.imsave(os.path.join(misclassified_output_folder, str(y_test[index]) + "_" + str(index)), image)