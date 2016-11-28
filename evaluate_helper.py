def evaluate_metric_functions(y_true, y_pred, metric_functions):
	"""Evaluates given metric functions on given true and predicted dense data.

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

def evaluate(name, x, y, predict_function):
  print("\nRunning evaluation: ", name)
  batch_size = BATCH_SIZE
  num_examples = x.shape[0]
  conf_matrix = np.zeros((2, 2))
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  loss_avg = 0
  for i in range(num_batches):
    batch_x = x[i*batch_size:(i+1)*batch_size, :]
    batch_y = y[i*batch_size:(i+1)*batch_size, :]
    logits_val= session.run([logits, loss], feed_dict={inputs:batch_x, labels:batch_y})
    yp = np.argmax(logits_val, 1)
    yt = np.argmax(batch_y, 1)
    conf_matrix_batch = confusion_matrix(yt, yp, labels=np.arange(2))
    np.add(conf_matrix, conf_matrix_batch, conf_matrix)
    loss_avg += loss_val
  loss_avg /= num_batches
  total_conf_matrix_sum = np.sum(conf_matrix)
  row_conf_matrix_sum = np.sum(conf_matrix, axis = 1)
  column_conf_matrix_sum = np.sum(conf_matrix, axis = 0)
  diagonal_conf_matrix_sum = np.sum(np.diag(conf_matrix))
  acc = diagonal_conf_matrix_sum / total_conf_matrix_sum
  prec = [conf_matrix[i][i] / column_conf_matrix_sum[i] for i in range(2)]
  rec = [conf_matrix[i][i] / row_conf_matrix_sum[i] for i in range(2)]
  print(name + " accuracy = %.2f" % acc)
  print(name + " per class precision = %s" % prec)
  print(name + " per class recall = %s" % rec)
  print(name + " avg loss = %.2f\n" % loss_avg)
  return loss_avg, acc