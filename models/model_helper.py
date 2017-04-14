import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


def dump_nparray(array, filename):
  #array_file = open(filename, 'w')
  array_file = open(filename, 'w')
  np.uint32(array.ndim).tofile(array_file)
  #for d in xrange(array.ndim):
  for d in range(array.ndim):
    np.uint32(array.shape[d]).tofile(array_file)
  array.tofile(array_file)
  array_file.close()


def load_nparray(filename, array_dtype):
  array_file = open(filename, 'r')
  n_dim = np.fromfile(array_file, dtype = np.uint32, count = 1)[0]
  shape = []
  #for d in xrange(n_dim):
  for d in range(n_dim):
    shape.append(np.fromfile(array_file, dtype = np.uint32, count = 1)[0])
  array_data = np.fromfile(array_file, dtype = array_dtype)
  array_file.close()
  return np.reshape(array_data, shape)


def _read_conv_params(in_dir, name):
  weights = load_nparray(in_dir + name + '_weights.bin', np.float32)
  biases = load_nparray(in_dir + name + '_biases.bin', np.float32)
  return weights, biases


def read_vgg_init(in_dir):
  names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
  layers = {}
  for name in names:
    weights, biases = _read_conv_params(in_dir, name)
    layers[name + '/weights'] = weights
    layers[name + '/biases'] = biases

  # transform fc6 parameters to conv6_1 parameters
  weights, biases = _read_conv_params(in_dir, 'fc6')
  weights = weights.reshape((7, 7, 512, 4096))
  layers['conv6_1' + '/weights'] = weights
  layers['conv6_1' + '/biases'] = biases
  names.append('conv6_1')
  return layers, names