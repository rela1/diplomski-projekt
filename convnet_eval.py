import os
import sys
import time
from shutil import copyfile

import numpy as np
import tensorflow as tf

import helper
import eval_helper
import train_helper
import attributes_dataset as dataset
from models import vgg_vertically_sliced

np.set_printoptions(linewidth=250)

BATCH_SIZE = 10
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-4
FULLY_CONNECTED = [200]
NUM_CLASSES = 2
EPOCHS = 150


def get_accuracy(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  predicted_labels = np.argmax(predictions, 1)
  assert predicted_labels.dtype == labels.dtype
  return 100.0 * np.sum(predicted_labels == labels) / predictions.shape[0]


def evaluate(sess, name, epoch_num, data_node, labels_node, logits, loss, data, labels):
  """ Trains the network
    Args:
      sess: TF session
      logits: network logits
  """
  print('\nPerformance on {}:'.format(name))
  loss_avg = 0
  data_size = data.shape[0]
  print('size = ', data_size)
  assert data_size % BATCH_SIZE == 0
  num_batches = data_size // BATCH_SIZE
  correct_cnt = 0
  for step in range(num_batches):
    offset = step * BATCH_SIZE
    batch_data = data[offset:(offset + BATCH_SIZE), ...]
    batch_labels = labels[offset:(offset + BATCH_SIZE)]
    feed_dict = {data_node: batch_data, labels_node: batch_labels}
    start_time = time.time()
    out_logits, loss_val = sess.run([logits, loss], feed_dict=feed_dict)
    duration = time.time() - start_time
    loss_avg += loss_val
    predicted_labels = out_logits.argmax(1)
    assert predicted_labels.dtype == batch_labels.dtype
    correct_cnt += np.sum(predicted_labels == batch_labels)
    if (step+1) % 10 == 0:
      num_examples_per_step = BATCH_SIZE
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)
      format_str = 'epoch %d, step %d / %d, loss = %.2f \
                    (%.1f examples/sec; %.3f sec/batch)'
      print(format_str % (epoch_num, step+1, num_batches, loss_val,
                          examples_per_sec, sec_per_batch))
  print('')
  accuracy = 100 * correct_cnt / data_size
  print('Accuracy = %.2f' % accuracy)
  return accuracy


def train(model, vgg_init_dir, dataset_root):
  """ Trains the network
  Args:
    model: module containing model architecture
  """
  print(vgg_init_dir)
  train_data = dataset.read_images(dataset_root, 'train').astype(np.float64)
  test_data = dataset.read_images(dataset_root, 'test').astype(np.float64)
  validate_data = dataset.read_images(dataset_root, 'validate').astype(np.float64)
  train_labels = dataset.read_labels(dataset_root, 'train').astype(np.int64)
  test_labels = dataset.read_labels(dataset_root, 'test').astype(np.int64)
  validate_labels = dataset.read_labels(dataset_root, 'validate').astype(np.int64)

  data_mean = train_data.reshape([-1, 3]).mean(0)
  data_std = train_data.reshape([-1, 3]).std(0)
  train_data = train_data.astype(np.float32)
  print('RGB mean = ', data_mean)
  print('RGB std = ', data_std)

  for c in range(train_data.shape[-1]):
    train_data[..., c] -= data_mean[c]
    test_data[..., c] -= data_mean[c]
    validate_data[..., c] -= data_mean[c]
    # better without variance normalization
    #train_data[..., c] /= data_std[c]
    #validate_data[..., c] /= data_std[c]
    #test_data[..., c] /= data_std[c]

  print(train_data.mean())
  print(train_data.std())
  print(train_data.flags['C_CONTIGUOUS'])
  print(train_labels.flags['C_CONTIGUOUS'])
  print(train_data.shape)
  print(train_labels.shape)

  train_size = train_data.shape[0]
  test_size = test_data.shape[0]
  validate_size = validate_data.shape[0]
  assert train_size % BATCH_SIZE == 0
  assert test_size % BATCH_SIZE == 0
  assert validate_size % BATCH_SIZE == 0

  with tf.Graph().as_default():
    sess = tf.Session()
    global_step = tf.get_variable('global_step', [], dtype=tf.int64,
        initializer=tf.constant_initializer(0), trainable=False)
    num_batches_per_epoch = train_size // BATCH_SIZE
    data_node = tf.placeholder(tf.float32,
        shape=(BATCH_SIZE, train_data.shape[1], train_data.shape[2], train_data.shape[3]))
    labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    with tf.variable_scope('model'):
      logits, loss, init_op, init_feed = model.build(data_node, labels_node, WEIGHT_DECAY, NUM_CLASSES, vgg_init_dir, fully_connected=FULLY_CONNECTED)
    with tf.variable_scope('model', reuse=True):
      logits_eval, loss_eval = model.build(data_node, labels_node, WEIGHT_DECAY, NUM_CLASSES, vgg_init_dir, is_training=False, fully_connected=FULLY_CONNECTED)
    exponential_learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 200, 0.96, staircase=True)
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
    saver.restore(sess, 'trained_models/best_convnet')
    evaluate(sess, 'train', 0, data_node, labels_node, logits_eval,
                          loss_eval, train_data, train_labels)
    evaluate(sess, 'validate', 0, data_node, labels_node, logits_eval,
                          loss_eval, validate_data, validate_labels)                    
    evaluate(sess, 'test', 0, data_node, labels_node, logits_eval,
                          loss_eval, test_data, test_labels)

if __name__ == '__main__':
  vgg_init_dir = sys.argv[1]
  dataset_root = sys.argv[2]
  train(vgg_vertically_sliced, vgg_init_dir, dataset_root)
