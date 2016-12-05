from sklearn import metrics
import time

METRIC_FUNCTIONS = (metrics.accuracy_score, metrics.precision_score, metrics.average_precision_score, metrics.recall_score)

def print_metrics(metrics):
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

def tf_predict_func(sess, inputs, logits):
	def predict_func(x):
		return np.argmax(sess.run(logits, feed_dict={inputs:x}))
	return predict_func

def evaluate(name, x, y, batch_size, predict_function):
  """
  Evaluates given predict function on given data in batches.

  Keyword arguments:
  name -- name of evaluation (train, test, valid, etc.)
  x -- data to be predicted
  y -- true densely stored classes
  batch_size -- size of evaluate batch
  predict_function -- function used for prediction; called as predict_function(x) on batch of data x

  Returns:
  dictionary mapping metric function name to metric score on given data with given predict function
  """
  print("\nRunning evaluation: ", name)
  num_examples = x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  y_predict = []
  for i in range(num_batches):
    batch_x = x[i*batch_size:(i+1)*batch_size, :]
    start_time = time.time()
    predict_batch_y = predict_function(batch_x)
    duration = time.time() - start_time
    y_predict.extend(predict_batch_y)
    if (i+1) % 10 == 0:
      print('step {}/{}, {} examples/sec, {} sec/batch'.format(i, num_batches, batch_size / duration, duration))
  metrics = evaluate_metric_functions(y, y_predict, METRIC_FUNCTIONS)
  print_metrics(metrics)
  return metrics