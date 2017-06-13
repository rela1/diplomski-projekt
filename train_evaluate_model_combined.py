import time
import math
import os
import shutil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from evaluate_helper import softmax, evaluate_default_metric_functions, print_metrics


np.set_printoptions(linewidth=250)


INFO_STEP = 5


def get_saver_variables():
  all_vars = tf.global_variables()
  filtered_vars = [var for var in all_vars if 'global_step' not in var.name and 'Adam' not in var.name and 'x___' not in var.name]
  filtered_vars_map = {var.name: var for var in filtered_vars}
  return filtered_vars_map


def get_restore_variables(model_path):
  reader = tf.train.NewCheckpointReader(model_path)
  saved_shapes = reader.get_variable_to_shape_map()
  all_vars = tf.global_variables()
  restore_vars = {var.name: var for var in all_vars if var.name in saved_shapes}
  return restore_vars


def train_model(fc_model, convolutional_model, dataset, sequence_length, num_epochs, learning_rate, model_path):
  sess = tf.Session()

  global_step = tf.get_variable('global_step', [], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)

  """
  opt = tf.train.AdamOptimizer(learning_rate)
  grads = opt.compute_gradients(fc_model.train_loss)
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train_op')
  """

  init_op, init_feed = fc_model.vgg_init

  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  sess.run(init_op, feed_dict=init_feed)

  saver = tf.train.Saver(get_saver_variables())

  writer = tf.summary.FileWriter(os.path.join(model_path, 'tensorboard'), sess.graph)
  print('\nTensorboard command: tensorboard --logdir="{}"'.format(os.path.abspath(os.path.join(model_path, 'tensorboard'))))
  writer.close()

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  mean_channels = dataset.mean_image_normalization(sess)

  best_valid_accuracy = 0.0

  positive_batch_labels = np.ones((dataset.batch_size, ), dtype=np.int32)
  batch_handle = 0
  for i in range(num_epochs):

    epoch_start_time = time.time()

    step = 0
    while True:
      start_time = time.time()

      batch_images, batch_masks, new_epoch, batch_handle = dataset.next_positive_batch(mean_channels, dataset.positive_sequences_dirs_train, batch_handle, dataset.batch_size)

      num_positive_examples = batch_images.shape[0] - sequence_length + 1

      for t in range(sequence_length - 1):
        representation_t = convolutional_model.spatials_train.forward(sess, t, batch_images[t])
        logits = convolutional_model.temporal_train.forward(sess, representation_t, positive_batch_labels, batch_masks[t])

      for t in range(sequence_length - 1, batch_images.shape[0]):
        representation_t = convolutional_model.spatials_train.forward(sess, t % sequence_length, batch_images[t])
        temporal_data = convolutional_model.temporal_train.forward_backward(sess, representation_t, positive_batch_labels, batch_masks[t])
        loss, cumulated_representation_gradient = temporal_data[0], temporal_data[2]
        convolutional_model.spatials_train.backward(sess, cumulated_representation_gradient[0], t % sequence_length - sequence_length + 1)

      for i in range(num_positive_examples):
        images = sess.run([dataset.train_images])
        for t in range(sequence_length - 1):
          representation_t = convolutional_model.spatials_train.forward(sess, t, batch_images[t])
          logits = convolutional_model.temporal_train.forward(sess, representation_t, positive_batch_labels, batch_masks[t])
          
        representation_t = convolutional_model.spatials_train.forward(sess, t % sequence_length, batch_images[t])
        temporal_data = convolutional_model.temporal_train.forward_backward(sess, representation_t, positive_batch_labels, batch_masks[t])
        loss, cumulated_representation_gradient = temporal_data[0], temporal_data[2]
        convolutional_model.spatials_train.backward(sess, cumulated_representation_gradient[0], t % sequence_length - sequence_length + 1)
      
      step += 1
      duration = time.time() - start_time

      assert not np.isnan(loss), 'Model diverged with loss = NaN'    

      if not step % INFO_STEP:
        print('\tEpoch: {}/{}, step loss: {}, {} examples/sec, learning rate: {}'.format(i+1, num_epochs, (loss + fc_loss) / 2, (batch_size * num_positive_examples * 2) / duration, learning_rate))

      if new_epoch:
        break

    epoch_duration = time.time() - epoch_start_time
    print('Done with epoch {}/{}, time needed: {}'.format(i + 1, num_epochs, epoch_duration))

    metrics, y_true, y_pred, y_prob = evaluate('Validation', sess, sequence_length, fc_model.valid_logits, fc_model.valid_loss, convolutional_model, dataset, dataset.num_valid_examples / 2, dataset.positive_sequences_dirs_valid, mean_channels)
    if metrics['accuracy_score'] > best_valid_accuracy:
      best_valid_accuracy = metrics['accuracy_score']
      print('\tNew best validation accuracy', best_valid_accuracy)
      saver.save(sess, model_path)

  print('Done training -- epoch limit reached')
  saver.restore(sess, model_path)
  evaluate('Test', sess, sequence_length, fc_model.test_logits, fc_model.test_loss, convolutional_model, dataset, dataset.num_test_examples / 2, dataset.positive_sequences_dirs_test, mean_channels)

  coord.request_stop()
  coord.join(threads)
  sess.close()


def evaluate(dataset_name, sess, sequence_length, fc_model_logits, fc_model_loss, convolutional_model, dataset, num_negative_examples, positives_sequences_dirs, mean_channels):
  print("\nRunning evaluation: ", dataset_name)
  y_true = []
  y_pred = []
  y_prob = []
  positive_batch_labels = np.ones((dataset.batch_size, ), dtype=np.int32)
  negative_batch_labels = np.zeros((dataset.batch_size, ), dtype=np.int32)
  batch_handle = 0
  step = 0
  print('Positive examples evaluation...')
  while True:

    start_time = time.time()

    batch_images, batch_masks, new_epoch, batch_handle = dataset.next_positive_batch(mean_channels, positives_sequences_dirs, batch_handle, dataset.batch_size)

    for t in range(sequence_length - 1):
      representation_t = convolutional_model.spatials_eval.forward(sess, t, batch_images[t])
      logits = convolutional_model.temporal_eval.forward(sess, representation_t, positive_batch_labels, batch_masks[t])

    for t in range(sequence_length - 1, batch_images.shape[0]):
      representation_t = convolutional_model.spatials_eval.forward(sess, t % sequence_length, batch_images[t])
      logits = convolutional_model.temporal_eval.forward(sess, representation_t, positive_batch_labels, batch_masks[t])
      preds = np.argmax(logits, axis=1)
      probs = softmax(logits)
      for i in range(dataset.batch_size):
        if batch_masks[t][0]:
          y_true.append(1)
          y_pred.append(preds[i])
          y_prob.append(probs[i])

    duration = time.time() - start_time

    if not step % 10:
      print('\tstep {}, {} examples/sec'.format(step + 1, (dataset.batch_size * (batch_images.shape[0] - sequence_length + 1)) / duration))

    step += 1

    if new_epoch:
      break

  num_batches = num_batches = int(math.ceil(num_negative_examples / dataset.batch_size))

  print('Negative examples evaluation...')
  for i in range(num_batches):
    start_time = time.time()
    logits_val, loss_val = sess.run([fc_model_logits, fc_model_loss])
    probs_val = softmax(logits_val)
    preds_val = np.argmax(logits_val, axis=1)
    y_pred.extend(preds_val)
    y_true.extend(negative_batch_labels)
    y_prob.extend(probs_val)
    duration = time.time() - start_time

    if not i % 10:
      print('\tstep {}/{}, {} examples/sec, {} sec/batch'.format(i+1, num_batches, dataset.batch_size / duration, duration))

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