import time
import math
import os
import shutil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utm

from evaluate_helper import evaluate, softmax, treshold_validate


np.set_printoptions(linewidth=250)


INFO_STEP = 20


def get_saver_variables():
  all_vars = tf.global_variables()
  filtered_vars = [var for var in all_vars if 'global_step' not in var.name and 'Adam' not in var.name and 'x___' not in var.name]
  filtered_vars_map = {var.name: var for var in filtered_vars}
  return filtered_vars_map


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


def freezed_pretrained_train_model(model, dataset, learning_rate, num_epochs, model_path):

  sess = get_session()

  trainable_variables = tf.trainable_variables()
  pretrained_variables = set(model.pretrained_vars)
  freezed_pretrained_trainable_variables = [var for var in trainable_variables if var not in pretrained_variables]

  global_step = tf.get_variable('global_step', [], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)

  opt = tf.train.AdamOptimizer(learning_rate)
  grads = opt.compute_gradients(model.train_loss, var_list=freezed_pretrained_trainable_variables)
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train_op')

  init_op, init_feed = model.vgg_init

  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  sess.run(init_op, feed_dict=init_feed)

  saver = tf.train.Saver(get_saver_variables())

  train_model(model, dataset, learning_rate, num_epochs, model_path, sess, global_step, train_op, saver)


def fine_tune_train_model(model, dataset, learning_rate, num_epochs, model_path):

  sess = get_session()

  global_step = tf.get_variable('global_step', [], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)

  opt = tf.train.AdamOptimizer(learning_rate)
  grads = opt.compute_gradients(model.train_loss)
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train_op')

  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
    
  saver = tf.train.Saver(get_saver_variables())
  saver.restore(sess, model_path)

  train_model(model, dataset, learning_rate, num_epochs, model_path, sess, global_step, train_op, saver, best_valid_evaluate=True)


def train_model(model, dataset, learning_rate, num_epochs, model_path, sess, global_step, train_op, saver, best_valid_evaluate=False, decay_learning_rate=True):

  num_batches = int(math.ceil(dataset.num_train_examples / dataset.batch_size))
  if decay_learning_rate:
    learning_rate = tf.train.exponential_decay(learning_rate, global_step, num_batches, 0.96)
  print('\nNumber of steps per epoch: {}'.format(num_batches))

  writer = tf.summary.FileWriter(os.path.join(model_path, 'tensorboard'), sess.graph)
  print('\nTensorboard command: tensorboard --logdir="{}"'.format(os.path.abspath(os.path.join(model_path, 'tensorboard'))))
  writer.close()

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  dataset.mean_image_normalization(sess)

  if best_valid_evaluate:
    metrics, y_true, y_pred, y_prob = evaluate('Validation', sess, model.valid_logits, model.valid_loss, dataset.valid_labels, dataset.num_valid_examples, dataset.batch_size)
    best_valid_accuracy = metrics['accuracy_score']
  else:
    best_valid_accuracy = 0.0

  operations = [train_op, model.train_loss]

  if decay_learning_rate:
    operations.append(learning_rate)

  for i in range(num_epochs):

    epoch_start_time = time.time()

    for j in range(num_batches):

      start_time = time.time()
      operations_results = sess.run(operations)
      loss_val = operations_results[1]
      learning_rate_val = operations_results[2] if decay_learning_rate else learning_rate
      duration = time.time() - start_time

      assert not np.isnan(loss_val), 'Model diverged with loss = NaN'    

      if not j % INFO_STEP:
        print('\tEpoch: {}/{}, step: {}/{}, loss: {}, {} examples/sec, {} sec/batch, learning rate: {}'.format(i+1, num_epochs, j+1, num_batches, loss_val, dataset.batch_size / duration, duration, learning_rate_val))

    epoch_duration = time.time() - epoch_start_time
    print('Done with epoch {}/{}, time needed: {}'.format(i + 1, num_epochs, epoch_duration))

    metrics, y_true, y_pred, y_prob = evaluate('Validation', sess, model.valid_logits, model.valid_loss, dataset.valid_labels, dataset.num_valid_examples, dataset.batch_size)
    if metrics['accuracy_score'] > best_valid_accuracy:
      best_valid_accuracy = metrics['accuracy_score']
      print('\tNew best validation accuracy', best_valid_accuracy)
      saver.save(sess, model_path)

  print('Done training -- epoch limit reached')
  saver.restore(sess, model_path)
  evaluate('Test', sess, model.test_logits, model.test_loss, dataset.test_labels, dataset.num_test_examples, dataset.batch_size)

  coord.request_stop()
  coord.join(threads)
  sess.close()


def evaluate_model(model, dataset, model_path):

  sess = get_session()

  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  saver = tf.train.Saver(get_saver_variables())
  saver.restore(sess, model_path)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  mean_channels = dataset.mean_image_normalization(sess)

  evaluate_dir_path = os.path.join(model_path, 'evaluation')
  shutil.rmtree(evaluate_dir_path)
  os.mkdir(evaluate_dir_path)
  
  metrics_dict, y_true, y_pred, y_prob = evaluate('Validation', sess, model.valid_logits, model.valid_loss, dataset.valid_labels, dataset.num_valid_examples, dataset.batch_size)
  tresholds, precisions, recalls, accuracies = treshold_validate(y_true, y_prob)
  argmax_accuracy = np.argmax(accuracies)
  validated_treshold = tresholds[argmax_accuracy]

  train_geolocations = dataset.train_geolocations if dataset.contains_geolocations else None
  evaluate_and_save_wrong_classifications('Train', sess, model.train_images, model.train_logits, model.train_loss, model.train_labels, train_geolocations, dataset.num_train_examples, dataset.batch_size, mean_channels, evaluate_dir_path, treshold=validated_treshold)
  
  valid_geolocations = dataset.valid_geolocations if dataset.contains_geolocations else None
  evaluate_and_save_wrong_classifications('Validation', sess, model.valid_images, model.valid_logits, model.valid_loss, model.valid_labels, valid_geolocations, dataset.num_valid_examples, dataset.batch_size, mean_channels, evaluate_dir_path, treshold=validated_treshold)

  test_geolocations = dataset.test_geolocations if dataset.contains_geolocations else None
  evaluate_and_save_wrong_classifications('Test', sess, model.test_images, model.test_logits, model.test_loss, model.test_labels, test_geolocations, dataset.num_test_examples, dataset.batch_size, mean_channels, evaluate_dir_path, treshold=validated_treshold)

  coord.request_stop()
  coord.join(threads)
  sess.close()