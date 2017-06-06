import os
import math

import tensorflow as tf
import numpy as np


class Dataset:

    def __init__(self, example_parser, dataset_suffix, dataset_root, batch_size, input_shape, is_training=True):
        self.batch_size = batch_size
        shapes = [input_shape, []]

        self.train_dir = os.path.join(dataset_root, 'train')
        self.valid_dir = os.path.join(dataset_root, 'validate')
        self.test_dir = os.path.join(dataset_root, 'test')

        self.train_tfrecords_dirs = [tfrecords_dir for tfrecords_dir in os.listdir(self.train_dir)]
        self.train_tfrecords = [os.path.join(self.train_dir, train_tfrecords_dir, train_tfrecords_dir + '_' + dataset_suffix + '.tfrecords') for train_tfrecords_dir in self.train_tfrecords_dirs]
        
        self.valid_tfrecords_dirs = [tfrecords_dir for tfrecords_dir in os.listdir(self.valid_dir)]
        self.valid_tfrecords = [os.path.join(self.valid_dir, valid_tfrecords_dir, valid_tfrecords_dir + '_' + dataset_suffix + '.tfrecords') for valid_tfrecords_dir in self.valid_tfrecords_dirs]

        self.test_tfrecords_dirs = [tfrecords_dir for tfrecords_dir in os.listdir(self.test_dir)]
        self.test_tfrecords = [os.path.join(self.test_dir, test_tfrecords_dir, test_tfrecords_dir + '_' + dataset_suffix + '.tfrecords') for test_tfrecords_dir in self.test_tfrecords_dirs]

        self.num_train_examples = number_of_examples(self.train_tfrecords_dirs, self.train_dir)
        self.num_valid_examples = number_of_examples(self.valid_tfrecords_dirs, self.valid_dir)
        self.num_test_examples = number_of_examples(self.test_tfrecords_dirs, self.test_dir)

        print('Train examples {}, validate examples {}, test examples {}'.format(self.num_train_examples, self.num_valid_examples, self.num_test_examples))

        train_file_queue = tf.train.string_input_producer(self.train_tfrecords, capacity=len(self.train_tfrecords))
        valid_file_queue = tf.train.string_input_producer(self.valid_tfrecords, capacity=len(self.valid_tfrecords))
        test_file_queue = tf.train.string_input_producer(self.test_tfrecords, capacity=len(self.test_tfrecords))

        train_images, train_labels = input_decoder(train_file_queue, example_parser)
        if is_training:
            self.train_images, self.train_labels = tf.train.shuffle_batch(
                [train_images, train_labels], batch_size=batch_size, shapes=shapes, capacity=100, min_after_dequeue=50)
        else:
            self.train_images, self.train_labels = tf.train.batch(
                [train_images, train_labels], batch_size=batch_size, shapes=shapes)

        valid_images, valid_labels = input_decoder(valid_file_queue, example_parser)
        self.valid_images, self.valid_labels = tf.train.batch(
            [valid_images, valid_labels], batch_size=batch_size, shapes=shapes)

        test_images, test_labels = input_decoder(test_file_queue, example_parser)
        self.test_images, self.test_labels = tf.train.batch(
            [test_images, test_labels], batch_size=batch_size, shapes=shapes)


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
            np.add(mean_image, np.sum(image_vals, axis=0), mean_image)
            del image_vals
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

    def mean_image_normalization(self, sess):
        num_batches = int(math.ceil(self.num_train_examples / self.batch_size))
        print('Mean image dataset normalization...')
        sequence_length = self.train_images.get_shape().as_list()[1]
        image_shape = self.train_images.get_shape().as_list()[2:]
        print('Image shape', image_shape)
        mean_image = np.zeros((image_shape))
        for i in range(num_batches):
            print('Normalization step {}/{}'.format(i + 1, num_batches))
            image_vals = sess.run(self.train_images)
            print(image_vals.shape)
            np.add(mean_image, np.sum(np.sum(image_vals, axis=0), axis=0), mean_image)
            del image_vals
        np.divide(mean_image, float(self.num_train_examples * sequence_length), mean_image)
        tf_mean_image = tf.constant(np.array([mean_image] * sequence_length), dtype=tf.float32)
        self.train_images = tf.subtract(self.train_images, tf_mean_image, name='train_images_mean_image_normalization')
        self.valid_images = tf.subtract(self.valid_images, tf_mean_image, name='valid_images_mean_image_normalization')
        self.test_images = tf.subtract(self.test_images, tf_mean_image, name='test_images_mean_image_normalization')
        print('Done with mean image dataset normalization...')


class ConvolutionalImageData(Dataset):

    def __init__(self, dataset_root, input_shape):
        super().__init__(parse_single_example, 'convolutional', dataset_root, 1, input_shape, is_training=False)
        self.num_positive_train_examples = number_of_examples(self.train_tfrecords_dirs, self.train_dir, 'positive_examples.txt')
        self.num_positive_valid_examples = number_of_examples(self.valid_tfrecords_dirs, self.valid_dir, 'positive_examples.txt')
        self.num_positive_test_examples = number_of_examples(self.test_tfrecords_dirs, self.test_dir, 'positive_examples.txt')

        print('Positive train examples {}, positive validate examples {}, positive test examples {}'.format(self.num_positive_train_examples, self.num_positive_valid_examples, self.num_positive_test_examples))



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


def number_of_examples(tfrecord_dirs, path_prefix, examples_file_name='examples.txt'):
  examples = 0
  for tfrecord_dir in tfrecord_dirs:
    with open(os.path.join(path_prefix, tfrecord_dir, examples_file_name)) as examples_file:
        examples += int(examples_file.read().strip())
  return examples