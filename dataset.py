import os
import math
from random import shuffle

import tensorflow as tf
import numpy as np
from matplotlib.image import imread


class Dataset:

    def __init__(self, example_parser, dataset_suffix, dataset_root, batch_size, input_shape, add_geolocations, is_training=True):
        self.batch_size = batch_size
        self.input_shape = input_shape

        if add_geolocations:
            shapes = [input_shape, [], [2]]
        else:
            shapes = [input_shape, []]

        self.contains_geolocations = add_geolocations

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

        if add_geolocations:
            train_images, train_labels, train_geolocations = input_decoder(train_file_queue, example_parser, add_geolocations)
            if is_training:
                self.train_images, self.train_labels, self.train_geolocations = tf.train.shuffle_batch(
                    [train_images, train_labels, train_geolocations], batch_size=batch_size, shapes=shapes, capacity=100, min_after_dequeue=50)
            else:
                self.train_images, self.train_labels, self.train_geolocations = tf.train.batch(
                    [train_images, train_labels, train_geolocations], batch_size=batch_size, shapes=shapes)
        else:
            train_images, train_labels = input_decoder(train_file_queue, example_parser, add_geolocations)
            if is_training:
                self.train_images, self.train_labels = tf.train.shuffle_batch(
                    [train_images, train_labels], batch_size=batch_size, shapes=shapes, capacity=100, min_after_dequeue=50)
            else:
                self.train_images, self.train_labels = tf.train.batch(
                    [train_images, train_labels], batch_size=batch_size, shapes=shapes)

        if add_geolocations:
            valid_images, valid_labels, valid_geolocations = input_decoder(valid_file_queue, example_parser, add_geolocations)
            self.valid_images, self.valid_labels, self.valid_geolocations = tf.train.batch(
                [valid_images, valid_labels, valid_geolocations], batch_size=batch_size, shapes=shapes)
        else:
            valid_images, valid_labels = input_decoder(valid_file_queue, example_parser, add_geolocations)
            self.valid_images, self.valid_labels = tf.train.batch(
                [valid_images, valid_labels], batch_size=batch_size, shapes=shapes)

        if add_geolocations:
            test_images, test_labels, test_geolocations = input_decoder(test_file_queue, example_parser, add_geolocations)
            self.test_images, self.test_labels, self.test_geolocations = tf.train.batch(
                [test_images, test_labels, test_geolocations], batch_size=batch_size, shapes=shapes)
        else:
            test_images, test_labels = input_decoder(test_file_queue, example_parser, add_geolocations)
            self.test_images, self.test_labels = tf.train.batch(
                [test_images, test_labels], batch_size=batch_size, shapes=shapes)


    def mean_image_normalization(self, sess):
        num_batches = int(math.ceil(self.num_train_examples / self.batch_size))
        print('Mean image dataset normalization...')
        image_shape = self.train_images.get_shape().as_list()[1:]
        print('Image shape', image_shape)
        mean_channels = np.zeros((3))
        for i in range(num_batches):
            print('Normalization step {}/{}'.format(i + 1, num_batches))
            image_vals = sess.run(self.train_images)
            mean_image_vals = np.mean(image_vals, axis=0)
            mean_image_channels = np.mean(mean_image_vals, axis=(0, 1))
            np.add(mean_channels, mean_image_channels, mean_channels)
        np.divide(mean_channels, float(num_batches), mean_channels)
        self.train_images = vgg_normalization(self.train_images, mean_channels)
        self.valid_images = vgg_normalization(self.valid_images, mean_channels)
        self.test_images = vgg_normalization(self.test_images, mean_channels)
        print('Done with mean channel image dataset normalization...')
        return mean_channels


class SingleImageDataset(Dataset):

    def __init__(self, dataset_root, batch_size, input_shape, add_geolocations, is_training=True):
        super().__init__(parse_single_example, 'single', dataset_root, batch_size, input_shape, add_geolocations, is_training=is_training)


class ImageSequenceDataset(Dataset):

    def __init__(self, dataset_root, batch_size, input_shape, add_geolocations, is_training=True):
        super().__init__(parse_sequence_example, 'sequential', dataset_root, batch_size, input_shape, add_geolocations, is_training=is_training)

    def mean_image_normalization(self, sess):
        num_batches = int(math.ceil(self.num_train_examples / self.batch_size))
        print('Mean image dataset normalization...')
        sequence_length = self.train_images.get_shape().as_list()[1]
        image_shape = self.train_images.get_shape().as_list()[2:]
        print('Image shape', image_shape)
        mean_channels = np.zeros((3))
        for i in range(num_batches):
            print('Normalization step {}/{}'.format(i + 1, num_batches))
            image_vals = sess.run(self.train_images)
            mean_image_vals = np.mean(image_vals, axis=(0, 1))
            mean_image_channels = np.mean(mean_image_vals, axis=(0, 1))
            np.add(mean_channels, mean_image_channels, mean_channels)
        np.divide(mean_channels, float(num_batches), mean_channels)
        self.train_images = vgg_normalization(self.train_images, mean_channels, axis=4)
        self.valid_images = vgg_normalization(self.valid_images, mean_channels, axis=4)
        self.test_images = vgg_normalization(self.test_images, mean_channels, axis=4)
        print('Done with mean image dataset normalization...')
        return mean_channels


class ConvolutionalImageSequenceDataset(Dataset):

    def __init__(self, dataset_root, input_shape):
        super().__init__(parse_single_example, 'convolutional', dataset_root, 1, input_shape, False, False)
        self.num_positive_train_examples = number_of_examples(self.train_tfrecords_dirs, self.train_dir, examples_file_name='positive_examples.txt')
        self.num_positive_valid_examples = number_of_examples(self.valid_tfrecords_dirs, self.valid_dir, examples_file_name='positive_examples.txt')
        self.num_positive_test_examples = number_of_examples(self.test_tfrecords_dirs, self.test_dir, examples_file_name='positive_examples.txt')


class CombinedImageSequenceDataset(Dataset):

    def __init__(self, dataset_root, batch_size, input_shape, is_training=True):
        super().__init__(parse_sequence_example, 'sequential', dataset_root, batch_size, input_shape, False, is_training=is_training)
        train_tfrecords_dirs = [os.path.join(self.train_dir, directory) for directory in self.train_tfrecords_dirs]
        valid_tfrecords_dirs = [os.path.join(self.valid_dir, directory) for directory in self.valid_tfrecords_dirs]
        test_tfrecords_dirs = [os.path.join(self.test_dir, directory) for directory in self.test_tfrecords_dirs]
        self.positive_sequences_dirs_train = self.get_sequences_dirs(train_tfrecords_dirs, 'positives')
        self.negative_sequences_dirs_train = self.get_sequences_dirs(train_tfrecords_dirs, 'negatives')
        self.positive_sequences_dirs_valid = self.get_sequences_dirs(valid_tfrecords_dirs, 'positives')
        self.negative_sequences_dirs_valid = self.get_sequences_dirs(valid_tfrecords_dirs, 'negatives')
        self.positive_sequences_dirs_test = self.get_sequences_dirs(test_tfrecords_dirs, 'positives')
        self.negative_sequences_dirs_test = self.get_sequences_dirs(test_tfrecords_dirs, 'negatives')

    def get_sequences_dirs(self, dirs, subset):
        subset_dirs = [os.path.join(_dir, subset) for _dir in dirs]
        sequences_dirs = []
        for subset_dir in subset_dirs:
            dir_sequences = os.listdir(subset_dir)
            sequences_dirs.extend([os.path.join(subset_dir, dir_sequence) for dir_sequence in dir_sequences])
        return sequences_dirs

    def mean_image_normalization(self, sess):
        num_batches = int(math.ceil((self.num_train_examples / 2) / self.batch_size))
        print('Mean image dataset normalization...')
        sequence_length = self.train_images.get_shape().as_list()[1]
        image_shape = self.train_images.get_shape().as_list()[2:]
        print('Image shape', image_shape)
        mean_channels = np.zeros((3))
        print('Normalizing negative examples...')
        negative_images_count = 0
        total_negative_steps = len(self.negative_sequences_dirs_train)
        for index, negative_sequence_dir in enumerate(self.negative_sequences_dirs_train):
            print('Normalization step {}/{}'.format(index + 1, total_negative_steps))
            for img_path in os.listdir(negative_sequence_dir):
                img_val = imread(os.path.join(negative_sequence_dir, img_path))[:, :, 0:3]
                mean_image_channels = np.mean(img_val, axis=(0, 1))
                negative_images_count += 1
                np.add(mean_channels, mean_image_channels, mean_channels)
        print('Normalizing positive examples...')
        positive_images_count = 0
        total_positive_steps = len(self.positive_sequences_dirs_train)
        for index, positive_sequence_dir in enumerate(self.positive_sequences_dirs_train):
            print('Normalization step {}/{}'.format(index + 1, total_positive_steps))
            for img_path in os.listdir(positive_sequence_dir):
                img_val = imread(os.path.join(positive_sequence_dir, img_path))[:, :, 0:3]
                mean_image_channels = np.mean(img_val, axis=(0, 1))
                positive_images_count += 1
                np.add(mean_channels, mean_image_channels, mean_channels)
        np.divide(mean_channels, float(negative_images_count + positive_images_count), mean_channels)
        print('Done with mean image dataset normalization...')
        return mean_channels

    def next_data_batch(self, mean_channels, sequences_dirs, last_batch_handle, batch_size):
        next_epoch = False
        if len(sequences_dirs) < batch_size * (last_batch_handle + 1):
            next_epoch = True
            shuffle(sequences_dirs)
            last_batch_handle = 0
        batch_dirs = sequences_dirs[last_batch_handle * batch_size : (last_batch_handle + 1) * batch_size]
        batch_sequence_length = 0
        for batch_dir in batch_dirs:
            batch_sequence_length = max(batch_sequence_length, len(os.listdir(batch_dir)))
        height = self.input_shape[1]
        width = self.input_shape[2]
        channels = self.input_shape[3]
        batch_images = np.zeros((batch_sequence_length, batch_size, height, width, channels), dtype=np.float32)
        batch_masks = np.zeros((batch_sequence_length, batch_size), dtype=np.float32)
        batch_axis = 0
        for batch_dir in batch_dirs:
            batch_dir_images = [os.path.join(batch_dir, image) for image in os.listdir(batch_dir)]
            batch_dir_images.sort()
            for index, batch_dir_image in enumerate(batch_dir_images):
                img = imread(batch_dir_image)[:, :, 0:3]
                img[:, :, 0] -= mean_channels[0]
                img[:, :, 1] -= mean_channels[1]
                img[:, :, 2] -= mean_channels[2]
                batch_images[index, batch_axis, :, :, :] = img
                batch_masks[index, batch_axis] = 1.0
            batch_axis += 1
            
        return batch_images, batch_masks, next_epoch, last_batch_handle + 1
        

def vgg_normalization(images, rgb_mean, axis=3):
    r, g, b = tf.split(axis=axis, num_or_size_splits=3, value=images)
    rgb = tf.concat(axis=axis, values=[
        r - rgb_mean[0],
        g - rgb_mean[1],
        b - rgb_mean[2]
    ])
    return rgb


def parse_sequence_example(record_string, add_geolocations):
  features_dict = {
                        'images_raw': tf.FixedLenFeature([], tf.string),
                        'label': tf.FixedLenFeature([], tf.int64),
                        'width' : tf.FixedLenFeature([], tf.int64),
                        'height' : tf.FixedLenFeature([], tf.int64),
                        'depth' : tf.FixedLenFeature([], tf.int64),
                        'sequence_length' : tf.FixedLenFeature([], tf.int64)
                    }
  if add_geolocations:
    features_dict['geo'] = tf.FixedLenFeature([], tf.string)
  features = tf.parse_single_example(
                    record_string,
                    features_dict
  )
  images = tf.decode_raw(features['images_raw'], tf.float32)
  width = tf.cast(features['width'], tf.int32)
  height = tf.cast(features['height'], tf.int32)
  depth = tf.cast(features['depth'], tf.int32)
  label = tf.cast(features['label'], tf.int32)
  sequence_length = tf.cast(features['sequence_length'], tf.int32)
  images = tf.reshape(images, [sequence_length, height, width, depth])
  if add_geolocations:
    geo = tf.decode_raw(features['geo'], tf.float32)
    geo = tf.reshape(geo, [2, ])
    return images, label, geo
  else:
    return images, label


def parse_single_example(record_string, add_geolocations):
    features_dict = {
                        'image_raw': tf.FixedLenFeature([], tf.string),
                        'label': tf.FixedLenFeature([], tf.int64),
                        'width' : tf.FixedLenFeature([], tf.int64),
                        'height' : tf.FixedLenFeature([], tf.int64),
                        'depth' : tf.FixedLenFeature([], tf.int64)
                    }
    if add_geolocations:
        features_dict['geo'] = tf.FixedLenFeature([], tf.string)
    features = tf.parse_single_example(
                    record_string,
                    features_dict
    )
    image = tf.decode_raw(features['image_raw'], tf.float32)
    width = tf.cast(features['width'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    image = tf.reshape(image, [height, width, depth])
    image = tf.cast(image, tf.float32)
    if add_geolocations:
        geo = tf.decode_raw(features['geo'], tf.float32)
        geo = tf.reshape(geo, [2, ])
        return image, label, geo
    else:
        return image, label


def input_decoder(filename_queue, example_parser, add_geolocations):
  reader = tf.TFRecordReader(
    options=tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.GZIP
    )
  )
  key, record_string = reader.read(filename_queue)
  return example_parser(record_string, add_geolocations)


def number_of_examples(tfrecord_dirs, path_prefix, examples_file_name='examples.txt'):
  examples = 0
  for tfrecord_dir in tfrecord_dirs:
    with open(os.path.join(path_prefix, tfrecord_dir, examples_file_name)) as examples_file:
        examples += int(examples_file.read().strip())
  return examples