import time
import math
import os
import shutil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from evaluate_helper import evaluate, softmax


np.set_printoptions(linewidth=250)


INFO_STEP = 20


def get_saver_variables():
  all_vars = tf.all_variables()
  filtered_vars = [var for var in all_vars if 'global_step' not in var.name and 'Adam' not in var.name]
  filtered_vars_map = {var.name: var for var in filtered_vars}
  return filtered_vars_map


def freezed_pretrained_train_model(model, dataset, learning_rate, num_epochs, model_path):

  sess = tf.Session()

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

  sess.run(tf.initialize_all_variables())
  sess.run(tf.initialize_local_variables())
  sess.run(init_op, feed_dict=init_feed)

  saver = tf.train.Saver(get_saver_variables())

  train_model(model, dataset, learning_rate, num_epochs, model_path, sess, global_step, train_op, saver)


def fine_tune_train_model(model, dataset, learning_rate, num_epochs, model_path):

  sess = tf.Session()

  global_step = tf.get_variable('global_step', [], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)

  opt = tf.train.AdamOptimizer(learning_rate)
  grads = opt.compute_gradients(model.train_loss)
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train_op')

  sess.run(tf.initialize_all_variables())
  sess.run(tf.initialize_local_variables())
    
  saver = tf.train.Saver(get_saver_variables())
  saver.restore(sess, model_path)

  train_model(model, dataset, learning_rate, num_epochs, model_path, sess, global_step, train_op, saver, best_valid_evaluate=True)


def train_model(model, dataset, learning_rate, num_epochs, model_path, sess, global_step, train_op, saver, best_valid_evaluate=False):

  num_batches = int(math.ceil(dataset.num_train_examples / dataset.batch_size))
  learning_rate = tf.train.exponential_decay(learning_rate, global_step, num_batches, 0.9, staircase=False)
  print('\nNumber of steps per epoch: {}'.format(num_batches))

  writer = tf.summary.FileWriter(os.path.join(model_path, 'tensorboard'), sess.graph)
  print('\nTensorboard command: tensorboard --logdir="{}"'.format(os.path.abspath(os.path.join(model_path, 'tensorboard'))))
  writer.close()

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  dataset.mean_image_normalization(sess, dataset.batch_size, num_batches)

  if best_valid_evaluate:
    metrics, y_true, y_pred, y_prob = evaluate('Validation', sess, model.valid_logits, model.valid_loss, dataset.valid_labels, dataset.num_valid_examples, dataset.batch_size)
    best_valid_accuracy = metrics['accuracy_score']
  else:
    best_valid_accuracy = 0.0

  losses = []
  correct = 0
  total = 0
  start_time = time.time()

  for i in range(num_epochs):

    for j in range(num_batches):

      start_time = time.time()
      _, loss_val, learning_rate_val = sess.run([train_op, model.train_loss, learning_rate])
      duration = time.time() - start_time

      assert not np.isnan(loss_val), 'Model diverged with loss = NaN'      

      if not j % INFO_STEP:
        print('\tEpoch: {}/{}, step: {}/{}, loss: {}, {} examples/sec, {} sec/batch, learning rate: {}'.format(i+1, num_epochs, j+1, num_batches, loss_val, dataset.batch_size / duration, duration, learning_rate_val))

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

  sess = tf.Session()

  sess.run(tf.initialize_all_variables())
  sess.run(tf.initialize_local_variables())
    
  saver = tf.train.Saver(get_saver_variables())
  saver.restore(sess, model_path)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  
  evaluate('Train', sess, model.train_logits, model.train_loss, dataset.train_labels, dataset.num_train_examples, dataset.batch_size)
  evaluate('Validation', sess, model.valid_logits, model.valid_loss, dataset.valid_labels, dataset.num_valid_examples, dataset.batch_size)
  evaluate('Test', sess, model.test_logits, model.test_loss, dataset.test_labels, dataset.num_test_examples, dataset.batch_size)

  coord.request_stop()
  coord.join(threads)
  sess.close()


def plot_wrong_classifications(model, dataset, model_path):

  sess = tf.Session()

  sess.run(tf.initialize_all_variables())
  sess.run(tf.initialize_local_variables())
    
  saver = tf.train.Saver()
  saver.restore(sess, model_path)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  for i in range(dataset.num_test_examples):

    logits_vals, label_vals, image_vals = sess.run([model.test_logits, dataset.test_labels, dataset.test_images])
    probability_vals = softmax(logits_vals)
    prediction_vals = np.argmax(logits_vals, axis=1)

    for j in range(dataset.batch_size):

      if label_vals[j] != prediction_vals[j]:

        if len(image_vals.shape) == 5:

          sequence_length = image_vals.shape[1]

          rows = 5
          cols = int(math.ceil(sequence_length / rows))

          for k in range(1, sequence_length + 1):
            plt.subplot(rows, cols, k)
            plt.imshow(image_vals[j][k - 1])

        else:

          plt.imshow(image_vals[j])

        plt.suptitle('True label {}, prediction: {}, probabilities: {}'.format(label_vals[j], prediction_vals[j], probability_vals[j]))
        plt.show()