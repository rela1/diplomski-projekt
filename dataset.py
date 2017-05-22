import os
import math

import tensorflow as tf
import numpy as np


class Dataset:

    def __init__(self, example_parser, dataset_suffix, dataset_root, batch_size, input_shape, is_training=True):
        self.batch_size = batch_size
        shapes = [input_shape, []]

        train_dir = os.path.join(dataset_root, 'train')
        valid_dir = os.path.join(dataset_root, 'validate')
        test_dir = os.path.join(dataset_root, 'test')

        train_tfrecords_dirs = [tfrecords_dir for tfrecords_dir in os.listdir(train_dir)]
        train_tfrecords = [os.path.join(train_dir, train_tfrecords_dir, train_tfrecords_dir + '_' + dataset_suffix + '.tfrecords') for train_tfrecords_dir in train_tfrecords_dirs]
        
        valid_tfrecords_dirs = [tfrecords_dir for tfrecords_dir in os.listdir(valid_dir)]
        valid_tfrecords = [os.path.join(valid_dir, valid_tfrecords_dir, valid_tfrecords_dir + '_' + dataset_suffix + '.tfrecords') for valid_tfrecords_dir in valid_tfrecords_dirs]

        test_tfrecords_dirs = [tfrecords_dir for tfrecords_dir in os.listdir(test_dir)]
        test_tfrecords = [os.path.join(test_dir, test_tfrecords_dir, test_tfrecords_dir + '_' + dataset_suffix + '.tfrecords') for test_tfrecords_dir in test_tfrecords_dirs]

        self.num_train_examples = number_of_examples(train_tfrecords_dirs)
        self.num_valid_examples = number_of_examples(valid_tfrecords_dirs)
        self.num_test_examples = number_of_examples(test_tfrecords_dirs)

        print('Train examples {}, validate examples {}, test examples {}'.format(self.num_train_examples, self.num_valid_examples, self.num_test_examples))

        train_file_queue = tf.train.string_input_producer(train_tfrecords)
        valid_file_queue = tf.train.string_input_producer(valid_tfrecords)
        test_file_queue = tf.train.string_input_producer(test_tfrecords)

        train_images, train_labels = input_decoder(train_file_queue, example_parser)
        if is_training:
            self.train_images, self.train_labels = tf.train.shuffle_batch(
                [train_images, train_labels], batch_size=batch_size, shapes=shapes, allow_smaller_final_batch=True, capacity=200, min_after_dequeue=100)
        else:
            self.train_images, self.train_labels = tf.train.batch(
                [train_images, train_labels], batch_size=batch_size, shapes=shapes, allow_smaller_final_batch=True)

        valid_images, valid_labels = input_decoder(valid_file_queue, example_parser)
        self.valid_images, self.valid_labels = tf.train.batch(
            [valid_images, valid_labels], batch_size=batch_size, shapes=shapes, allow_smaller_final_batch=True)

        test_images, test_labels = input_decoder(test_file_queue, example_parser)
        self.test_images, self.test_labels = tf.train.batch(
            [test_images, test_labels], batch_size=batch_size, shapes=shapes, allow_smaller_final_batch=True)

    def mean_image_normalization(self, sess):
        num_batches = int(math.ceil(self.num_train_examples / self.batch_size))
        print('Mean image dataset normalization...')
        image_shape = self.train_images.get_shape().as_list()[1:]
        print('Image shape', image_shape)
        mean_image = np.zeros((image_shape))
        for i in range(num_batches):
            print('Normalization step {}/{}'.format(i + 1, num_batches))
            image_vals = sess.run(self.train_images)
            print(image_vals.shape)
            for j in range(len(image_vals)):
                np.add(mean_image, image_vals[j], mean_image)
        np.divide(mean_image, float(self.num_train_examples), mean_image)
        tf_mean_image = tf.constant(mean_image, dtype=tf.float32)
        self.train_images = tf.subtract(self.train_images, tf_mean_image, name='train_images_mean_image_normalization')
        self.valid_images = tf.subtract(self.valid_images, tf_mean_image, name='valid_images_mean_image_normalization')
        self.test_images = tf.subtract(self.test_images, tf_mean_image, name='test_images_mean_image_normalization')
        print('Done with mean image dataset normalization...')


class SingleImageDataset(Dataset):

    def __init__(self, dataset_root, batch_size, input_shape, is_training=True):
        super().__init__(parse_single_example, 'single', dataset_root, batch_size, input_shape, is_training=is_training)


class ImageSequenceDataset(Dataset):

    def __init__(self, dataset_root, batch_size, input_shape, is_training=True):
        super().__init__(parse_sequence_example, 'sequential', dataset_root, batch_size, input_shape, is_training=is_training)


def parse_sequence_example(record_string):
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


def parse_single_example(record_string):
    features = tf.parse_single_example(
                    record_string,
                    features={
                        'image_raw': tf.FixedLenFeature([], tf.string),
                        'label': tf.FixedLenFeature([], tf.int64),
                        'width' : tf.FixedLenFeature([], tf.int64),
                        'height' : tf.FixedLenFeature([], tf.int64),
                        'depth' : tf.FixedLenFeature([], tf.int64)
                    }
    )
    image = tf.decode_raw(features['image_raw'], tf.float32)
    width = tf.cast(features['width'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    image = tf.reshape(image, [height, width, depth])
    return image, label


def input_decoder(filename_queue, example_parser):
  reader = tf.TFRecordReader(
    options=tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.GZIP
    )
  )
  key, record_string = reader.read(filename_queue)
  return example_parser(record_string)


def number_of_examples(tfrecord_dirs):
  examples = 0
  for tfrecord_dir in tfrecord_dirs:
    with open(os.path.join(tfrecord_dir, 'examples.txt')) as examples_file:
        examples += int(examples_file.read().strip())
  return examples