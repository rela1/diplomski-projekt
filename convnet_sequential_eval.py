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


def evaluate(name, sess, logits, loss, labels, num_examples):
  print("\nRunning evaluation: ", name)
  y_true = []
  y_pred = []
  losses = []
  num_batches = int(math.ceil(num_examples / BATCH_SIZE))
  for i in range(num_batches):
    logit_val, loss_val, labels_val = sess.run([logits, loss, labels])
    pred = np.argmax(logit_val, axis=1)
    y_pred.extend(pred)
    y_true.extend(labels_val)
    losses.append(loss_val)
  metrics = evaluate_default_metric_functions(y_true, y_pred)
  print_metrics(metrics)
  print('\taverage loss={}\n'.format(np.mean(losses)))
  return metrics


def evaluate_trained_model(model, dataset_root, model_path):
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

    train_file_queue = tf.train.string_input_producer(train_tfrecords, num_epochs=EPOCHS)
    valid_file_queue = tf.train.string_input_producer(valid_tfrecords)
    test_file_queue = tf.train.string_input_producer(test_tfrecords)

    print('Train tfrecords: {}, valid tfrecords: {}, test tfrecords: {}'.format(train_tfrecords, valid_tfrecords, test_tfrecords))
    print('Train num examples: {}, valid num examples: {}, test num examples: {}'.format(train_examples, valid_examples, test_examples))

    train_images, train_labels = input_decoder(train_file_queue)
    train_images, train_labels = tf.train.batch(
        [train_images, train_labels], batch_size=BATCH_SIZE, shapes=SHAPES, allow_smaller_final_batch=True)

    valid_images, valid_labels = input_decoder(valid_file_queue)
    valid_images, valid_labels = tf.train.batch(
        [valid_images, valid_labels], batch_size=BATCH_SIZE, shapes=SHAPES, allow_smaller_final_batch=True)

    test_images, test_labels = input_decoder(test_file_queue)
    test_images, test_labels = tf.train.batch(
      [test_images, test_labels], batch_size=BATCH_SIZE, shapes=SHAPES, allow_smaller_final_batch=True)

    train_input_placeholder = tf.placeholder_with_default(train_images, shape=[None] + INPUT_SHAPE)
    train_label_placeholder = tf.placeholder_with_default(train_labels, shape=(None, ))

    valid_input_placeholder = tf.placeholder_with_default(valid_images, shape=[None] + INPUT_SHAPE)
    valid_label_placeholder = tf.placeholder_with_default(valid_labels, shape=(None, ))

    test_input_placeholder = tf.placeholder_with_default(test_images, shape=[None] + INPUT_SHAPE)
    test_label_placeholder = tf.placeholder_with_default(test_labels, shape=(None, ))

    sess = tf.Session()
    global_step = tf.get_variable('global_step', [], dtype=tf.int64,
        initializer=tf.constant_initializer(0), trainable=False)

    with tf.variable_scope('model'):
      logit_train, loss_train = model.build_sequential(train_input_placeholder, train_label_placeholder, fully_connected=FULLY_CONNECTED, is_training=False)
      logit_valid, loss_valid = model.build_sequential(valid_input_placeholder, valid_label_placeholder, fully_connected=FULLY_CONNECTED, is_training=False)
      logit_test, loss_test = model.build_sequential(test_input_placeholder, test_label_placeholder, fully_connected=FULLY_CONNECTED, is_training=False)

    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    evaluate('Train', sess, logit_train, loss_train, train_labels, train_examples)
    evaluate('Validate', sess, logit_valid, loss_valid, valid_labels, valid_examples)
    evaluate('Test', sess, logit_test, loss_test, test_labels, test_examples)

    coord.request_stop()
    coord.join(threads)
    sess.close()

    best_accuracy = 0
    saver = tf.train.Saver()

    losses = []
    correct = 0
    total = 0
    step = 0
    best_valid_accuracy = 0
    start_time = time.time()
    try:
      while not coord.should_stop():

          _, logits_val, loss_val, labels_val = sess.run([train_op, logit, loss, train_labels])
          correct += np.sum(np.argmax(logits_val, axis=1) == labels_val)
          total += BATCH_SIZE
          assert not np.isnan(loss_val), 'Model diverged with loss = NaN'
          losses.append(loss_val)
          step += 1

          if not step % INFO_STEP:
            duration = time.time() - start_time
            print('Average loss: {}, average accuracy: {}, examples/sec: {}, sec/step: {}'.format(np.mean(losses), correct / total, INFO_STEP * BATCH_SIZE / duration, float(duration)))
            start_time = time.time()
            losses.clear()
            correct = 0
            total = 0

          if (step * BATCH_SIZE) >= train_examples:
            step = 0
            metrics = evaluate('Validation', sess, logit_eval, loss_eval, valid_labels, valid_examples)
            if metrics['accuracy_score'] > best_valid_accuracy:
              best_valid_accuracy = metrics['accuracy_score']
              print('\tNew best validation accuracy', best_valid_accuracy)
              saver.save(sess, model_path)
    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
      saver.restore(sess, model_path)
      evaluate('Test', sess, logit_test, loss_test, test_labels, test_examples)
    finally:
      coord.request_stop()

    coord.join(threads)
    sess.close()


if __name__ == '__main__':
  dataset_root = sys.argv[1]
  model_path = sys.argv[2]
  evaluate_trained_model(vgg_vertically_sliced, dataset_root, model_path)