import os
import sys
import time

import numpy as np
import tensorflow as tf

from models import vgg_vertically_sliced
from evaluate_helper import evaluate_default_metric_functions, print_metrics

np.set_printoptions(linewidth=250)

WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-3
FULLY_CONNECTED = [200]
EPOCHS = 30
INFO_STEP = 20
BATCH_SIZE = 5
INPUT_SHAPE = [25, 40, 100, 3]
SHAPES = [INPUT_SHAPE, []]


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
  images = tf.decode_raw(features['images_raw'], tf.float32)
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


def evaluate(name, sess, logits, loss, tf_records_files, input_placeholder, label_placeholder):
  print("\nRunning evaluation: ", name)
  y_true = []
  y_pred = []
  losses = []
  for tf_records_file in tf_records_files:
    for record_string in tf.python_io.tf_record_iterator(tf_records_file, options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)):
        images, labels = parse_example(record_string)
        images_val, labels_val = sess.run([images, labels])
        logit_val, loss_val = sess.run([logits, loss], feed_dict={input_placeholder: [images_val], label_placeholder: [labels_val]})
        pred = np.argmax(logit_val, axis=1)
        y_pred.append(pred)
        y_true.append(labels_val)
        losses.append(loss_val)
  metrics = evaluate_default_metric_functions(y_true, y_pred)
  print_metrics(metrics)
  print('\taverage loss={}'.format(np.mean(losses)))
  return metrics


def evaluate(model, dataset_root, model_path):
  train_dir = os.path.join(dataset_root, 'train')
  valid_dir = os.path.join(dataset_root, 'validate')
  test_dir = os.path.join(dataset_root, 'test')

  train_examples = number_of_examples(train_dir)
  valid_examples = number_of_examples(valid_dir)
  test_examples = number_of_examples(test_dir)

  with tf.Graph().as_default():

    train_tfrecords = [os.path.join(train_dir, file) for file in os.listdir(train_dir)]
    valid_tfrecords = [os.path.join(valid_dir, file) for file in os.listdir(valid_dir)]
    test_tfrecords = [os.path.join(test_dir, file) for file in os.listdir(test_dir)]

    input_placeholder = tf.placeholder(tf.float32, shape=[None] + INPUT_SHAPE)
    label_placeholder = tf.placeholder(tf.int64, shape=(None, ))

    sess = tf.Session()
    global_step = tf.get_variable('global_step', [], dtype=tf.int64,
        initializer=tf.constant_initializer(0), trainable=False)

    with tf.variable_scope('model'):
      logit_eval, loss_eval = model.build_sequential(input_placeholder, label_placeholder, fully_connected=FULLY_CONNECTED, is_training=False)

    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    evaluate('Train', sess, logit_eval, loss_eval, train_tfrecords, input_placeholder, label_placeholder)
    evaluate('Validate', sess, logit_eval, loss_eval, valid_tfrecords, input_placeholder, label_placeholder)
    evaluate('Test', sess, logit_eval, loss_eval, test_tfrecords, input_placeholder, label_placeholder)
    
    sess.close()


if __name__ == '__main__':
  dataset_root = sys.argv[1]
  model_path = sys.argv[2]
  evaluate(vgg_vertically_sliced, dataset_root, model_path)