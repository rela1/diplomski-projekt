import os

import tensorflow as tf


class Dataset:

    def __init__(self, example_parser, dataset_root, batch_size, input_shape, is_training=True):
        self.batch_size = batch_size
        shapes = [input_shape, []]

        train_dir = os.path.join(dataset_root, 'train')
        valid_dir = os.path.join(dataset_root, 'validate')
        test_dir = os.path.join(dataset_root, 'test')

        self.num_train_examples = number_of_examples(train_dir)
        self.num_valid_examples = number_of_examples(valid_dir)
        self.num_test_examples = number_of_examples(test_dir)

        train_tfrecords = [os.path.join(train_dir, file) for file in os.listdir(train_dir)]
        valid_tfrecords = [os.path.join(valid_dir, file) for file in os.listdir(valid_dir)]
        test_tfrecords = [os.path.join(test_dir, file) for file in os.listdir(test_dir)]

        train_file_queue = tf.train.string_input_producer(train_tfrecords)
        valid_file_queue = tf.train.string_input_producer(valid_tfrecords)
        test_file_queue = tf.train.string_input_producer(test_tfrecords)

        train_images, train_labels = input_decoder(train_file_queue, example_parser)
        if is_training:
            self.train_images, self.train_labels = tf.train.shuffle_batch(
                [train_images, train_labels], batch_size=batch_size, shapes=shapes, allow_smaller_final_batch=True, capacity=50000, min_after_dequeue=10000)
        else:
            self.train_images, self.train_labels = tf.train.batch(
                [train_images, train_labels], batch_size=batch_size, shapes=shapes, allow_smaller_final_batch=True)

        valid_images, valid_labels = input_decoder(valid_file_queue, example_parser)
        self.valid_images, self.valid_labels = tf.train.batch(
            [valid_images, valid_labels], batch_size=batch_size, shapes=shapes, allow_smaller_final_batch=True)

        test_images, test_labels = input_decoder(test_file_queue, example_parser)
        self.test_images, self.test_labels = tf.train.batch(
            [test_images, test_labels], batch_size=batch_size, shapes=shapes, allow_smaller_final_batch=True)


class SingleImageDataset(Dataset):

    def __init__(self, dataset_root, batch_size, input_shape, is_training=True):
        super().__init__(parse_single_example, dataset_root, batch_size, input_shape, is_training=is_training):


class ImageSequenceDataset(Dataset):

    def __init__(self, dataset_root, batch_size, input_shape, is_training=True):
        super().__init__(parse_sequence_example, dataset_root, batch_size, input_shape, is_training=is_training):


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
    return images, label


def input_decoder(filename_queue, example_parser):
  reader = tf.TFRecordReader(
    options=tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.GZIP
    )
  )
  key, record_string = reader.read(filename_queue)
  return example_parser(record_string)


def number_of_examples(directory):
  examples = 0
  for fn in [os.path.join(directory, file) for file in os.listdir(directory)]:
    for record in tf.python_io.tf_record_iterator(fn, options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)):
      examples += 1
  return examples