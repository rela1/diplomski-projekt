import os
import sys
import time

import numpy as np
import tensorflow as tf

from models import vgg_vertically_sliced
from evaluate_helper import evaluate_default_metric_functions, print_metrics

np.set_printoptions(linewidth=250)

WEIGHT_DECAY = 1e-3
LEARNING_RATE = 0.25
FULLY_CONNECTED = [400, 50]
EPOCHS = 5
INFO_STEP = 20
BATCH_SIZE = 2
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
        images, label = parse_example(record_string)
        images_val, label_val = sess.run([images, label])
        logit_val, loss_val = sess.run([logits, loss], feed_dict={input_placeholder: images_val, label_placeholder: [label_val]})
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

  with tf.Graph().as_default():

    train_tfrecords = [os.path.join(train_dir, file) for file in os.listdir(train_dir)]
    valid_tfrecords = [os.path.join(valid_dir, file) for file in os.listdir(valid_dir)]
    test_tfrecords = [os.path.join(test_dir, file) for file in os.listdir(test_dir)]

    train_file_queue = tf.train.string_input_producer(train_tfrecords, num_epochs=EPOCHS)

    print('Train tfrecords: {}, valid tfrecords: {}, test tfrecords: {}'.format(train_tfrecords, valid_tfrecords, test_tfrecords))
    print('Train num examples: {}, valid num examples: {}, test num examples: {}'.format(train_examples, valid_examples, test_examples))

    train_images, train_labels = input_decoder(train_file_queue)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * 1
    train_images, train_labels = tf.train.shuffle_batch(
        [train_images, train_labels], batch_size=BATCH_SIZE, capacity=capacity,
        min_after_dequeue=min_after_dequeue, shapes=SHAPES)

    input_placeholder = tf.placeholder_with_default(train_images, shape=[BATCH_SIZE] + INPUT_SHAPE)
    label_placeholder = tf.placeholder_with_default(train_labels, shape=(BATCH_SIZE, ))

    sess = tf.Session()
    global_step = tf.get_variable('global_step', [], dtype=tf.int64,
        initializer=tf.constant_initializer(0), trainable=False)

    with tf.variable_scope('model'):
      logit, loss, init_op, init_feed = model.build_sequential(input_placeholder, label_placeholder, fully_connected=FULLY_CONNECTED, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir, is_training=True)
    with tf.variable_scope('model', reuse=True):
      logit_eval, loss_eval = model.build_sequential(input_placeholder, label_placeholder, fully_connected=FULLY_CONNECTED, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir, is_training=False)

    opt = tf.train.AdamOptimizer(LEARNING_RATE)
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
      train_op = tf.no_op(name='train')

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
    try:
      while not coord.should_stop():

          _, logits_val, loss_val, labels_val = sess.run([train_op, logit, loss, train_labels])
          print(logits_val)
          correct += np.sum(np.argmax(logits_val, axis=1) == label_val)
          total += BATCH_SIZE
          assert not np.isnan(loss_val), 'Model diverged with loss = NaN'
          losses.append(loss_val)
          step += 1

          if not step % INFO_STEP:
            duration = time.time() - start_time
            print('Average loss: {}, average accuracy: {}, examples/sec: {}, sec/step: {}'.format(np.mean(losses), correct / total, INFO_STEP / duration, float(duration)))
            start_time = time.time()
            losses.clear()
            correct = 0
            total = 0

          if not step % train_examples:
            metrics = evaluate('Validate', sess, logit_eval, loss_eval, valid_tfrecords, input_placeholder, label_placeholder)
            if metrics['accuracy_score'] > best_valid_accuracy:
              best_valid_accuracy = metrics['accuracy_score']
              print('New best validation accuracy', best_valid_accuracy)
              saver.save(sess, model_path)
    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
      saver.restore(sess, model_path)
      evaluate('Test', sess, logit_eval, loss_eval, test_tfrecords, input_placeholder, label_placeholder)
    finally:
      coord.request_stop()

    coord.join(threads)
    sess.close()


if __name__ == '__main__':
  vgg_init_dir = sys.argv[1]
  dataset_root = sys.argv[2]
  model_path = sys.argv[3]
  train(vgg_vertically_sliced, vgg_init_dir, dataset_root, model_path)