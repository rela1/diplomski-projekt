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

if __name__ == '__main__':
	X_test_left = dataset.read_convolutional_features(sys.argv[1], sys.argv[3], 'test')
	X_test_middle = dataset.read_convolutional_features(sys.argv[1], sys.argv[4], 'test')
	X_test = np.concatenate((X_test_left, X_test_middle), axis=1)
	y_test = dataset.read_labels(sys.argv[1], 'test')

	model = joblib.load(sys.argv[5])
	y_test_pred = model.predict(X_test)
	y_test_score = model.decision_function(X_test)
	y_test_score_sort = np.argsort(y_test_score)
	test_metrics = evaluate_helper.evaluate_metric_functions(y_test, y_test_pred, METRIC_FUNCTIONS)
	print("Test data results:")
	print_metrics(test_metrics)

	X_test_imgs = dataset.read_images(sys.argv[2], 'test')
	misclassified_output_folder = sys.argv[6]
	for index, image in enumerate(X_test_imgs):
		if y_test_pred[index] != y_test[index]:
			img.imsave(os.path.join(misclassified_output_folder, str(y_test[index]) + "_" + str(y_test_score[index])) + ".png", image)

	for index, image in enumerate(X_test_imgs):
		plt.imshow(image)
		plt.title('Real class={}, predicted class={}'.format(y_test[index], y_test_pred[index]))
		plt.show()