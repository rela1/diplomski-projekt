import math
import time

import numpy as np
from sklearn import metrics
import tensorflow as tf

METRIC_FUNCTIONS = (metrics.accuracy_score, metrics.precision_score, metrics.average_precision_score, metrics.recall_score)


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


def evaluate_default_metric_functions(y_true, y_pred):
  """
  Evaluates given metric functions on given true and predicted dense data.

  Keyword arguments:
  y_true -- true densely stored classes
  y_pred -- predicted densely stored classes

  Returns:
  dictionary mapping metric function name to metric score
  """
  return evaluate_metric_functions(y_true, y_pred, METRIC_FUNCTIONS)


def evaluate(name, sess, batch_logits, batch_loss, batch_true_labels, num_examples, batch_size):
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
  """
  print("\nRunning evaluation: ", name)
  y_true = []
  y_pred = []
  y_prob = []
  losses = []
  num_batches = int(math.ceil(num_examples / batch_size))
  for i in range(num_batches):
    start_time = time.time()
    logits_val, loss_val, labels_val = sess.run([logits, loss, labels])
    duration = time.time() - start_time
    probs_val = softmax()
    preds_val = np.argmax(logits_val, axis=1)
    y_pred.extend(preds_val)
    y_true.extend(labels_val)
    y_prob.extend(probs_val)
    losses.append(loss_val)
    if not i % 10:
      print('\tstep {}/{}, {} examples/sec, {} sec/batch'.format(i+1, num_batches, batch_size / duration, duration))
  metrics = evaluate_default_metric_functions(y_true, y_pred)
  print_metrics(metrics)
  print('\taverage loss={}\n'.format(np.mean(losses)))
  return metrics, y_true, y_pred, y_prob