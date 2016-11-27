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