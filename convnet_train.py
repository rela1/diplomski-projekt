import os
import sys
import time

import numpy as np
import tensorflow as tf

import attributes_dataset as dataset
from models import vgg_vertically_sliced
import evaluate_helper

np.set_printoptions(linewidth=250)

BATCH_SIZE = 10
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-4
FULLY_CONNECTED = []
NUM_CLASSES = 2
EPOCHS = 150

def train(model, vgg_init_dir, dataset_root, model_path):
  train_data, train_labels, validate_data, validate_labels, test_data, test_labels = dataset.read_and_normalize_images(dataset_root)

  train_size = train_data.shape[0]

  with tf.Graph().as_default():
    sess = tf.Session()
    global_step = tf.get_variable('global_step', [], dtype=tf.int64,
        initializer=tf.constant_initializer(0), trainable=False)
    num_batches_per_epoch = train_size // BATCH_SIZE

    data_node = tf.placeholder(tf.float32,
        shape=(BATCH_SIZE, train_data.shape[1], train_data.shape[2], train_data.shape[3]))
    labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))

    with tf.variable_scope('model'):
      logits, loss, init_op, init_feed = model.build(data_node, labels_node, NUM_CLASSES, fully_connected=FULLY_CONNECTED, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir)
    with tf.variable_scope('model', reuse=True):
      logits_eval, loss_eval = model.build(data_node, labels_node, NUM_CLASSES, fully_connected=FULLY_CONNECTED, is_training=False)

    exponential_learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 200, 0.99, staircase=True)
    opt = tf.train.AdamOptimizer(exponential_learning_rate)
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
      train_op = tf.no_op(name='train')

    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    sess.run(init_op, feed_dict=init_feed)

    ex_start_time = time.time()
    num_batches = train_size // BATCH_SIZE
    global_step_val = 0
    best_accuracy = 0
    saver = tf.train.Saver()
    for epoch_num in range(1, EPOCHS + 1):
      indices = np.arange(train_size)
      np.random.shuffle(indices)
      train_data = np.ascontiguousarray(train_data[indices])
      train_labels = np.ascontiguousarray(train_labels[indices])
      for step in range(num_batches):
        offset = step * BATCH_SIZE 
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        feed_dict = {data_node: batch_data, labels_node: batch_labels}
        start_time = time.time()
        run_ops = [train_op, loss, logits, global_step]
        _, loss_val, scores, global_step_val = sess.run(run_ops, feed_dict=feed_dict)
        duration = time.time() - start_time
        assert not np.isnan(loss_val), 'Model diverged with loss = NaN'
        if (step+1) % 5 == 0:
          num_examples_per_step = BATCH_SIZE
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)
          format_str = 'epoch %d, batch %d / %d, loss = %.2f \
            (%.1f examples/sec; %.3f sec/batch)'
          print(format_str % (epoch_num, step+1, num_batches, loss_val, examples_per_sec, sec_per_batch))
      valid_accuracy = evaluate_helper.evaluate('validate', validate_data, validate_labels, BATCH_SIZE, 
      	evaluate_helper.tf_predict_func(sess, data_node, logits_eval), verbose=True)['accuracy_score']
      if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        saver.save(sess, model_path)
      print('Best validate accuracy = %.2f' % best_accuracy)
    saver.restore(sess, model_path)
    evaluate_helper.evaluate('test', test_data, test_labels, BATCH_SIZE, 
      	evaluate_helper.tf_predict_func(sess, data_node, logits_eval), verbose=True)

if __name__ == '__main__':
  vgg_init_dir = sys.argv[1]
  dataset_root = sys.argv[2]
  model_path = sys.argv[3]
  train(vgg_vertically_sliced, vgg_init_dir, dataset_root, model_path)
