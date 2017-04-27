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


def train_model(model, dataset, learning_rate, num_epochs, model_path):

  sess = tf.Session()

  global_step = tf.get_variable('global_step', [], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)
  #learning_rate = tf.train.exponential_decay(learning_rate, global_step, 5000, 0.5, staircase=True)

  opt = tf.train.AdamOptimizer(learning_rate)
  grads = opt.compute_gradients(model.train_loss)
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train')

  if os.path.isdir(os.path.abspath(os.path.join(model_path, 'tensorboard'))):
    shutil.rmtree(os.path.abspath(os.path.join(model_path, 'tensorboard')))

  print('Variables list:')
  print([x.name for x in tf.global_variables()])

  writer = tf.summary.FileWriter(os.path.join(model_path, 'tensorboard'), sess.graph)
  print('Tensorboard command: tensorboard --logdir="{}"'.format(os.path.abspath(os.path.join(model_path, 'tensorboard'))))
  writer.close()

  init_op, init_feed = model.vgg_init

  sess.run(tf.initialize_all_variables())
  sess.run(tf.initialize_local_variables())
  sess.run(init_op, feed_dict=init_feed)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  best_accuracy = 0
  saver = tf.train.Saver()

  losses = []
  correct = 0
  total = 0
  step = 0
  best_valid_accuracy = 0
  start_time = time.time()
  
  num_batches = int(math.ceil(dataset.num_train_examples / dataset.batch_size))

  for i in range(num_epochs):

    for j in range(num_batches):

      start_time = time.time()
      _, loss_val = sess.run([train_op, model.train_loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_val), 'Model diverged with loss = NaN'      

      if not step % INFO_STEP:
        print('\tEpoch: {}/{}, step: {}/{}, loss: {}, {} examples/sec, {} sec/batch'.format(i+1, num_epochs, j+1, num_batches, loss_val, dataset.batch_size / duration, duration))

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
    
  saver = tf.train.Saver()
  saver.restore(sess, model_path)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  
  evaluate('Train', sess, model.train_logits, model.train_loss, dataset.train_labels, dataset.num_train_examples, dataset.batch_size)
  evaluate('Validation', sess, model.valid_logits, model.valid_loss, dataset.valid_labels, dataset.num_valid_examples, dataset.batch_size)
  evaluate('Test', sess, model.test_logits, model.test_loss, dataset.test_labels, dataset.num_test_examples, dataset.batch_size)


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