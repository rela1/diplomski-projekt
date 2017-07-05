import math
import time
import shutil
import os

import numpy as np
from sklearn import metrics
import tensorflow as tf
from matplotlib.image import imsave

METRIC_FUNCTIONS = (metrics.accuracy_score, metrics.precision_score, metrics.recall_score)
PROBABILITY_METRIC_FUNCTIONS = (metrics.average_precision_score, )


def print_metrics(metrics):
  print('Results:')
  for metric in metrics:
    print('\t{}={}'.format(metric, metrics[metric]))


def softmax(x):
  probs = np.exp(x)
  probs_sum = np.sum(probs, axis=1, keepdims=True)
  return probs / probs_sum


def evaluate_metric_functions(y_true, y_pred, metric_functions):
  """
  Evaluates given metric functions on given true and predicted dense data.

  Keyword arguments:
  y_true -- true densely stored classes
  y_pred -- predicted densely stored classes
  metric_functions -- iterable of metric_functions which receive 2 arguments: y_true and y_pred and returns metric score

  Returns:
  dictionary mapping metric function name to metric score
  """
  return {metric_function.__name__ : metric_function(y_true, y_pred) for metric_function in metric_functions}


def evaluate_default_metric_functions(y_true, y_pred, y_prob):
  """
  Evaluates given metric functions on given true and predicted dense data.

  Keyword arguments:
  y_true -- true densely stored classes
  y_pred -- predicted densely stored classes and probabilities
  y_prob -- predicted densely stored probabilities

  Returns:
  dictionary mapping metric function name to metric score
  """
  pred_metrics = evaluate_metric_functions(y_true, y_pred, METRIC_FUNCTIONS)
  pred_metrics.update(evaluate_metric_functions(y_true, y_prob[:, 1], PROBABILITY_METRIC_FUNCTIONS))
  return pred_metrics


def evaluate(name, sess, batch_logits, batch_loss, batch_true_labels, num_examples, batch_size, treshold=0.5):
  """
  Evaluates given logits and loss on true labels. Data is expected to be read from TFRecords queue reader.

  Keyword arguments:
  name -- name of evaluation subset (train, test, valid, etc.)
  sess -- session used for evaluation
  batch_logits -- logits of batch data that is to be evaluated
  batch_loss -- mean loss of batch data that is to be evaluated
  batch_true_labels -- true labels of batch data that is to be evaluated
  num_examples -- total number of examples in data that is to be evaluated
  batch_size -- batch size of data that is to be evaluated
  treshold -- treshold for classifying example as positive

  Returns:
  metrics_dict -- dictionary mapping metric function name to metric score
  y_true -- true predictions
  y_pred -- model output predictions
  y_prob -- model output probabilities
  """
  print("\nRunning evaluation: ", name)
  y_true = []
  y_prob = []
  losses = []
  num_batches = int(math.ceil(num_examples / batch_size))
  for i in range(num_batches):
    start_time = time.time()
    logits_val, loss_val, labels_val = sess.run([batch_logits, batch_loss, batch_true_labels])
    duration = time.time() - start_time
    probs_val = softmax(logits_val)
    y_true.extend(labels_val)
    y_prob.extend(probs_val)
    losses.append(loss_val)
    if not i % 10:
      print('\tstep {}/{}, {} examples/sec, {} sec/batch'.format(i+1, num_batches, batch_size / duration, duration))
  y_prob = np.array(y_prob)
  y_pred = np.zeros(len(y_prob), dtype=np.int8)
  y_pred[y_prob[:, 1] >= treshold] = 1
  metrics_dict = evaluate_default_metric_functions(y_true, y_pred, y_prob)
  print_metrics(metrics_dict)
  print('\taverage loss={}\n'.format(np.mean(losses)))
  cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
  print('\tconfusion matrix=\n{}\n'.format(cm))
  return metrics_dict, y_true, y_pred, y_prob


def evaluate_and_save_wrong_classifications(name, sess, batch_images, batch_logits, batch_loss, batch_true_labels, batch_geolocations, num_examples, batch_size, mean_channels, save_path, treshold=0.5):
  """
  Evaluates given logits and loss on true labels. Data is expected to be read from TFRecords queue reader.
  All images that are classified wrong are saved on given path along with geolocations if provided.

  Keyword arguments:
  name -- name of evaluation subset (train, test, valid, etc.)
  sess -- session used for evaluation
  batch_images -- batch of input images
  batch_logits -- logits of batch data that is to be evaluated
  batch_loss -- mean loss of batch data that is to be evaluated
  batch_true_labels -- true labels of batch data that is to be evaluated
  batch_geolocations -- geolocations of batch data that is to be evaluated
  num_examples -- total number of examples in data that is to be evaluated
  batch_size -- batch size of data that is to be evaluated
  mean_channels -- per channel mean of train images
  save_path -- save path for saving images that are classified wrong and for saving geolocations if provided
  treshold -- treshold for classifying example as positive
  """
  print("\nRunning evaluation: ", name)
  y_true = []
  y_prob = []
  y_pred = []
  losses = []

  num_batches = int(math.ceil(num_examples / batch_size))
  operations = [batch_images, batch_logits, batch_true_labels, batch_loss]

  if batch_geolocations is not None:
    operations.append(batch_geolocations)
    image_name_to_geolocation = {}

  false_positives_dir = os.path.join(save_path, name, 'false_positives')
  if os.path.isdir(false_positives_dir):
    shutil.rmtree(false_positives_dir)
  os.makedirs(false_positives_dir)
  false_negatives_dir = os.path.join(save_path, name, 'false_negatives')
  if os.path.isdir(false_negatives_dir):
    shutil.rmtree(false_negatives_dir)
  os.makedirs(false_negatives_dir)

  img_cnt = 0

  for i in range(num_batches):
    start_time = time.time()
    results = sess.run(operations)
    duration = time.time() - start_time

    image_vals = results[0]
    logits_vals = results[1]
    labels_vals = results[2]
    loss_val = results[3]

    probs_vals = softmax(logits_vals)
    preds_vals = np.zeros(len(probs_vals), dtype=np.int8)
    preds_vals[probs_vals[:, 1] >= treshold] = 1

    y_true.extend(labels_vals)
    y_prob.extend(probs_vals)
    y_pred.extend(preds_vals)
    losses.append(loss_val)

    if batch_geolocations is not None:
      geolocations = results[4]

    for j in range(batch_size):

      if labels_vals[j] != preds_vals[j]:

        if len(image_vals.shape) == 5:
          image_val = image_vals[j, -1]
        else:
          image_val = image_vals[j]

        np.add(image_val, mean_channels, image_val)

        if preds_vals[j] == 1:
          img_name = os.path.join(false_positives_dir, str(img_cnt) + '.png')
        else:
          img_name = os.path.join(false_negatives_dir, str(img_cnt) + '.png')

        if batch_geolocations is not None:
          image_name_to_geolocation[img_name] = geolocations[j]

        imsave(img_name, image_val)
        img_cnt += 1

    if not i % 10:
      print('\tstep {}/{}, {} examples/sec, {} sec/batch'.format(i+1, num_batches, batch_size / duration, duration))

  y_prob = np.array(y_prob)
  metrics_dict = evaluate_default_metric_functions(y_true, y_pred, y_prob)
  print_metrics(metrics_dict)
  print('\taverage loss={}\n'.format(np.mean(losses)))
  cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
  print('\tconfusion matrix=\n{}\n'.format(cm))

  if batch_geolocations is not None:
    with open(os.path.join(save_path, name, 'geo_data.txt'), 'w') as f:
      for image_name in image_name_to_geolocation:
        geolocation = image_name_to_geolocation[image_name]
        f.write(image_name + ' -> ' + str(geolocation[0]) + ',' + str(geolocation[1]) + '\n')


def treshold_validate(y_true, y_prob, n_tresholds=100):
  """
  Calculates precision, recall and accuracy performance values for n_tresholds different tresholds.

  Keyword arguments:
  y_true -- true predictions
  y_prob -- model output probabilities
  n_tresholds -- number of different tresholds to use for calculating the performance values

  Returns:
  tresholds -- tresholds used for calculating the performance values
  precisions -- precisions for different values of tresholds
  recalls -- recalls for different values of tresholds
  accuracies -- accuracies for different values of tresholds
  """
  tresholds = np.linspace(0, 1, n_tresholds)
  precisions = []
  recalls = []
  accuracies = []
  for treshold in tresholds:
    y_pred = np.zeros(len(y_prob), dtype=np.int8)
    y_pred[y_prob[:, 1] >= treshold] = 1
    precisions.append(metrics.precision_score(y_true, y_pred))
    recalls.append(metrics.recall_score(y_true, y_pred))
    accuracies.append(metrics.accuracy_score(y_true, y_pred))
  return tresholds, np.array(precisions), np.array(recalls), np.array(accuracies)