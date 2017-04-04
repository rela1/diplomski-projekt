import os
import sys
import time

import numpy as np
import tensorflow as tf

from models import vgg_vertically_sliced
from evaluate_helper import evaluate_default_metric_functions

np.set_printoptions(linewidth=250)

WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-4
FULLY_CONNECTED = [200]
EPOCHS = 1
INFO_STEP = 20


def parse_example(record_string):
  features = tf.parse_single_example(
                    record_string,
                    features={
                        'images_raw': tf.FixedLenFeature([], tf.string),
                        'label': tf.FixedLenFeature([], tf.int64),
                        'width' : tf.FixedLenFeature([], tf.int64),
                        'height' : tf.FixedLenFeature([], tf.int64),
                        'depth' : tf.FixedLenFeature([], tf.int64),
                        'sequence_length' : tf.FixedLenFeature([], tf.int64)
                    }
  )
  images = tf.decode_raw(features['images_raw'], tf.float64)
  width = tf.cast(features['width'], tf.int32)
  height = tf.cast(features['height'], tf.int32)
  depth = tf.cast(features['depth'], tf.int32)
  label = tf.cast(features['label'], tf.int32)
  sequence_length = tf.cast(features['sequence_length'], tf.int32)
  images = tf.reshape(images, [sequence_length, height, width, depth])
  return images, label


def input_decoder(filename_queue):
  reader = tf.TFRecordReader(
    options=tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.GZIP
    )
  )
  key, record_string = reader.read(filename_queue)
  return parse_example(record_string)


def number_of_examples(directory):
  examples = 0
  for fn in [os.path.join(directory, file) for file in os.listdir(directory)]:
    for record in tf.python_io.tf_record_iterator(fn, options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)):
      examples += 1
  return examples


def evaluate(name, sess, logit, loss, label, num_examples):
  print("\nRunning evaluation: ", name)
  y_true = []
  y_pred = []
  losses = []
  for i in range(num_examples):
    logit_val, loss_val, label_val = sess.run([logit, loss, label])
    pred = np.argmax(logit_val, axis=1)
    y_pred.append(pred)
    y_true.append(label_val)
    losses.append(loss_val)
  metrics = evaluate_default_metric_functions(y_true, y_pred)
  print_metrics(metrics)
  print('\taverage loss={}'.format(np.mean(losses)))
  return metrics




def train(model, vgg_init_dir, dataset_root, model_path):
  train_dir = os.path.join(dataset_root, 'train')
  valid_dir = os.path.join(dataset_root, 'validate')
  test_dir = os.path.join(dataset_root, 'test')

  train_examples = number_of_examples(train_dir)
  valid_examples = number_of_examples(valid_dir)
  test_examples = number_of_examples(test_dir)

  train_file_queue = tf.train.string_input_producer([os.path.join(train_dir, file) for file in os.listdir(train_dir)], num_epochs=EPOCHS)
  valid_file_queue = tf.train.string_input_producer([os.path.join(valid_dir, file) for file in os.listdir(valid_dir)])
  test_file_queue = tf.train.string_input_producer([os.path.join(test_dir, file) for file in os.listdir(test_dir)])

  train_images, train_label = input_decoder(train_file_queue)
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  train_images, train_label = tf.train.shuffle_batch(
      [train_images, train_label], batch_size=1, capacity=capacity,
      min_after_dequeue=min_after_dequeue)

  valid_images, valid_label = input_decoder(valid_file_queue)

  test_images, test_label = input_decoder(test_file_queue)

  with tf.Graph().as_default():
    sess = tf.Session()
    global_step = tf.get_variable('global_step', [], dtype=tf.int64,
        initializer=tf.constant_initializer(0), trainable=False)

    with tf.variable_scope('model'):
      logit, loss, init_op, init_feed = model.build(data_node, labels_node, NUM_CLASSES, fully_connected=FULLY_CONNECTED, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir)
    with tf.variable_scope('model', reuse=True):
      test_logit_eval, test_loss_eval = model.build_sequential(test_images, test_label, fully_connected=FULLY_CONNECTED, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir, is_training=False)
      valid_logit_eval, valid_loss_eval = model.build_sequential(valid_images, valid_label, fully_connected=FULLY_CONNECTED, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir, is_training=False)

    exponential_learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 2000, 0.5, staircase=True)
    opt = tf.train.AdamOptimizer(exponential_learning_rate)
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
      train_op = tf.no_op(name='train')

    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    sess.run(init_op, feed_dict=init_feed)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ex_start_time = time.time()
    global_step_val = 0
    best_accuracy = 0
    saver = tf.train.Saver()

    losses = []
    step = 0
    best_valid_accuracy = 0
    try:
      start_time = time.time()
      while not coord.should_stop():

          _, loss_val = sess.run([train_op, loss])
          assert not np.isnan(loss_val), 'Model diverged with loss = NaN'
          losses.append(loss_val)
          step += 1

          if not step % INFO_STEP:
            duration = time.time() - start_time
            print('Average loss: {}, examples/sec: {}, sec/step: {}'.format(np.mean(losses), INFO_STEP / duration, float(duration)))
            start_time = time.time()

          if not step % train_examples:
            metrics = evaluate(sess, valid_logit_eval, valid_loss_eval, valid_label, valid_examples)
            if metrics['accuracy_score'] > best_valid_accuracy:
              best_valid_accuracy = metrics['accuracy_score']
              saver.save(sess, model_path)
    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
      saver.restore(sess, model_path)
      evaluate(sess, test_logit_eval, test_loss_eval, test_label, test_examples)
    finally:
      coord.request_stop()

    coord.join(threads)
    sess.close()


if __name__ == '__main__':
  vgg_init_dir = sys.argv[1]
  dataset_root = sys.argv[2]
  model_path = sys.argv[3]
  train(vgg_vertically_sliced, vgg_init_dir, dataset_root, model_path)
