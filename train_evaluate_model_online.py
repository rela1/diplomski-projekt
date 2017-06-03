import time
import math
import os
import shutil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from evaluate_helper import softmax, evaluate_default_metric_functions, print_metrics


np.set_printoptions(linewidth=250)


INFO_STEP = 20


def get_saver_variables():
  all_vars = tf.global_variables()
  filtered_vars = [var for var in all_vars if 'global_step' not in var.name and 'Adam' not in var.name]
  filtered_vars_map = {var.name: var for var in filtered_vars}
  return filtered_vars_map


def get_restore_variables(model_path):
  reader = tf.train.NewCheckpointReader(model_path)
  saved_shapes = reader.get_variable_to_shape_map()
  all_vars = tf.global_variables()
  restore_vars = {var.name: var for var in all_vars if var.name in saved_shapes}
  return restore_vars


def train_model(model, dataset, sequence_length, num_epochs, learning_rate, pretrained_model_path, model_path):
  sess = tf.Session()

  global_step = tf.get_variable('global_step', [], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)

  init_op, init_feed = model.vgg_init

  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  sess.run(init_op, feed_dict=init_feed)

  """
  restore_vars = get_restore_variables(pretrained_model_path)
  loader = tf.train.Saver(restore_vars)
  loader.restore(sess, pretrained_model_path)
  """

  saver = tf.train.Saver(get_saver_variables())

  writer = tf.summary.FileWriter(os.path.join(model_path, 'tensorboard'), sess.graph)
  print('\nTensorboard command: tensorboard --logdir="{}"'.format(os.path.abspath(os.path.join(model_path, 'tensorboard'))))
  writer.close()

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  dataset.mean_image_normalization(sess)

  metrics, y_true, y_pred, y_prob = evaluate('Validation', sess, sequence_length, model.spatials_valid, model.temporal_valid, dataset.num_valid_examples)
  best_valid_accuracy = metrics['accuracy_score']

  for i in range(num_epochs):

    for t in range(sequence_length - 1):
      representation_t = model.spatials_train.forward(sess, t)
      logits = model.temporal_train.forward(sess, representation_t)

    for t in range(sequence_length - 1, dataset.num_train_examples):
      start_time = time.time()
      representation_t = model.spatials_train.forward(sess, t % sequence_length)
      temporal_data = model.temporal_train.forward_backward(sess, representation_t)
      loss, cumulated_representation_gradient = temporal_data[0], temporal_data[2]
      model.spatials_train.backward(sess, [cumulated_representation_gradient[0]], t % sequence_length - sequence_length + 1)
      duration = time.time() - start_time

      assert not np.isnan(loss), 'Model diverged with loss = NaN'    

      if not t % INFO_STEP:
        print('\tEpoch: {}/{}, step: {}/{}, loss: {}, {} examples/sec, {} sec/batch, learning rate: {}'.format(i+1, num_epochs, t, dataset.num_train_examples, loss, 1 / duration, duration, learning_rate))

    metrics, y_true, y_pred, y_prob = evaluate('Validation', sess, sequence_length, model.spatials_valid, model.temporal_valid, dataset.num_valid_examples)
    if metrics['accuracy_score'] > best_valid_accuracy:
      best_valid_accuracy = metrics['accuracy_score']
      print('\tNew best validation accuracy', best_valid_accuracy)
      saver.save(sess, model_path)

  print('Done training -- epoch limit reached')
  saver.restore(sess, model_path)
  evaluate('Test', sess, sequence_length, model.spatials_test, model.temporal_test, dataset.num_test_examples)

  coord.request_stop()
  coord.join(threads)
  sess.close()


def evaluate(dataset_name, sess, sequence_length, spatials_model, temporal_model, number_of_examples):
  print("\nRunning evaluation: ", dataset_name)
  y_true = []
  y_pred = []
  y_prob = []
  for t in range(sequence_length - 1):
    representation_t = spatials_model.forward(sess, t)
    logits_val, labels_val = temporal_model.forward(sess, representation_t)
    preds_val = np.argmax(logits_val, axis=1)
    probs_val = softmax(logits_val)
    y_pred.extend(preds_val)
    y_true.extend(labels_val)
    y_prob.extend(probs_val)
  for t in range(sequence_length - 1, number_of_examples):
    start_time = time.time()
    representation_t = spatials_model.forward(sess, t % sequence_length)
    logits_val, labels_val = temporal_model.forward(sess, representation_t)
    preds_val = np.argmax(logits_val, axis=1)
    probs_val = softmax(logits_val)
    y_pred.extend(preds_val)
    y_true.extend(labels_val)
    y_prob.extend(probs_val)
    duration = time.time() - start_time
    if not t % 10:
      print('\tstep {}/{}, {} examples/sec, {} sec/batch'.format(t+1, number_of_examples, 1 / duration, duration))
  metrics = evaluate_default_metric_functions(y_true, y_pred)
  print_metrics(metrics)
  return metrics, y_true, y_pred, y_prob


def evaluate_model(model, dataset, model_path):

  sess = tf.Session()

  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  saver = tf.train.Saver(get_saver_variables())
  saver.restore(sess, model_path)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  dataset.mean_image_normalization(sess)
  
  evaluate('Train', sess, model.train_logits, model.train_loss, dataset.train_labels, dataset.num_train_examples, dataset.batch_size)
  evaluate('Validation', sess, model.valid_logits, model.valid_loss, dataset.valid_labels, dataset.num_valid_examples, dataset.batch_size)
  evaluate('Test', sess, model.test_logits, model.test_loss, dataset.test_labels, dataset.num_test_examples, dataset.batch_size)

  coord.request_stop()
  coord.join(threads)
  sess.close()