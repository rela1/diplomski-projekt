import numpy as np
from sklearn import metrics
import time

METRIC_FUNCTIONS = (metrics.accuracy_score, metrics.precision_score, metrics.average_precision_score, metrics.recall_score)

def print_metrics(metrics):
  print('Results:')
  for metric in metrics:
    print('\t{}={}'.format(metric, metrics[metric]))

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

def tf_proba_predict_func(probabilities):
  return np.argmax(probabilities, axis=1)

def tf_probability_func(sess, inputs, logits):
  def probability_func(x):
    logits_val = sess.run(logits, feed_dict={inputs:x})
    logits_exp = np.exp(logits_val)
    logits_exp_sum = np.sum(logits_exp, axis=1, keepdims=True)
    return logits_exp / logits_exp_sum

def evaluate(name, x, y, batch_size, predict_function, probability_function=None, verbose=False):
  """
  Evaluates given predict function on given data in batches.

  Keyword arguments:
  name -- name of evaluation (train, test, valid, etc.)
  x -- data to be predicted
  y -- true densely stored classes
  batch_size -- size of evaluate batch
  predict_function -- function used for prediction; called as predict_function(x) on batch of data x
  probability_function -- function used for probability calculation; called as probability_function(x) on batch of data x; if probability functon is given, predict function is called on batch of probabilities of data x instead of batch of data x

  Returns:
  dictionary mapping metric function name to metric score on given data with given predict function
  """
  print("\nRunning evaluation: ", name)
  num_examples = x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  y_predict = []
  if not probability_function == None:
    y_probabilities = []
  for i in range(num_batches):
    batch_x = x[i*batch_size:(i+1)*batch_size, :]
    start_time = time.time()
    if not probability_function == None:
      probabilities_batch_y = probability_function(batch_x)
      y_probabilities.extend(probabilities_batch_y)
      predict_batch_y = predict_function(probabilities_batch_y)
    else:
      predict_batch_y = predict_function(batch_x)
    duration = time.time() - start_time
    y_predict.extend(predict_batch_y)
    if (i+1) % 10 == 0 and verbose:
      print('step {}/{}, {} examples/sec, {} sec/batch'.format(i+1, num_batches, batch_size / duration, duration))
  metrics = evaluate_metric_functions(y, y_predict, METRIC_FUNCTIONS)
  print_metrics(metrics)
  if not probability_function == None:
    return metrics, y_predict, y_probabilities
  else:
    return metrics, y_predict