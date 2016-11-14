import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from models.model_helper import read_vgg_init
from tensorflow.contrib.layers.python.layers import initializers

import losses

FLAGS = tf.app.flags.FLAGS

def total_loss_sum(losses):
  # Assemble all of the losses for the current tower only.
  # Calculate the total loss for the current tower.
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
  return total_loss


def create_init_op(vgg_layers):
  variables = tf.contrib.framework.get_variables()
  init_map = {}
  for var in variables:
    name_split = var.name.split('/')
    if len(name_split) != 3:
      continue
    name = name_split[1] + '/' + name_split[2][:-2]
    if name in vgg_layers:
      print(var.name, ' --> init from ', name)
      init_map[var.name] = vgg_layers[name]
    else:
      print(var.name, ' --> random init')
  init_op, init_feed = tf.contrib.framework.assign_from_values(init_map)
  return init_op, init_feed


def build_convolutional_feature_extractor(inputs, weight_decay, vgg_init_dir, is_training=True, vertical_slice=None):
  vgg_layers, vgg_layer_names = read_vgg_init(vgg_init_dir)

  weight_decay = FLAGS.weight_decay

  with tf.contrib.framework.arg_scope([layers.convolution2d],
      kernel_size=3, stride=1, padding='SAME', rate=1, activation_fn=tf.nn.relu,
      normalizer_fn=None, weights_initializer=None,
      weights_regularizer=layers.l2_regularizer(weight_decay)):

    net = layers.convolution2d(inputs, 64, scope='conv1_1')
    net = layers.convolution2d(net, 64, scope='conv1_2')
    net = layers.max_pool2d(net, 2, 2, scope='pool1')
    net = layers.convolution2d(net, 128, scope='conv2_1')
    net = layers.convolution2d(net, 128, scope='conv2_2')
    net = layers.max_pool2d(net, 2, 2, scope='pool2')
    net = layers.convolution2d(net, 256, scope='conv3_1')
    net = layers.convolution2d(net, 256, scope='conv3_2')
    net = layers.convolution2d(net, 256, scope='conv3_3')
    net = layers.max_pool2d(net, 2, 2, scope='pool3')
    net = layers.convolution2d(net, 512, scope='conv4_1')
    net = layers.convolution2d(net, 512, scope='conv4_2')
    net = layers.convolution2d(net, 512, scope='conv4_3')
    net = layers.max_pool2d(net, 2, 2, scope='pool4')
    net = layers.convolution2d(net, 512, scope='conv5_1')
    net = layers.convolution2d(net, 512, scope='conv5_2')
    net = layers.convolution2d(net, 512, scope='conv5_3')
    net = layers.max_pool2d(net, 2, 2, scope='pool5')

    net = layers.convolution2d(net, 4096, kernel_size=7, scope='conv6_1')
    
    if vertical_slice != None:
      last_convolution_filter_size = net.get_shape()
      net = layers.max_pool2d(net, kernel_size=[int(last_convolution_filter_size[1]), 1], stride=1)
      slice_size = int(int(last_convolution_filter_size[2]) / 3)
      net = tf.slice(net, begin=[0, 0, vertical_slice * slice_size, 0], size=[-1, -1, slice_size, -1])

    net = tf.contrib.layers.flatten(net, scope='flatten')

    if is_training:
      init_op, init_feed = create_init_op(vgg_layers)
      return net, init_op, init_feed

    return net, None, None

def build(inputs, labels, weight_decay, vgg_init_dir, is_training=True, vertical_slice=None, ):

  # to big weight_decay = 5e-3
  bn_params = {
      # Decay for the moving averages.
      'decay': 0.999,
      'center': True,
      'scale': True,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # None to force the updates
      'updates_collections': None,
      'is_training': is_training,
  }

  net, init_op, init_feed = build_convolutional_feature_extractor(inputs, weight_decay, vgg_init_dir, is_training, vertical_slice)

  with tf.contrib.framework.arg_scope([layers.fully_connected],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_initializer=initializers.xavier_initializer(),
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    net = layers.fully_connected(net, 10, scope='fc7')
  logits = layers.fully_connected(net, FLAGS.num_classes, activation_fn=None,
      weights_regularizer=layers.l2_regularizer(weight_decay), scope='logits')

  total_loss = loss(logits, labels, is_training)
  
  if is_training:
    return logits, total_loss, init_op, init_feed

  return logits, total_loss

def loss(logits, labels, is_training):
  xent_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
  total_loss = total_loss_sum([xent_loss])
  if is_training:
    loss_averages_op = losses.add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
      total_loss = tf.identity(total_loss)

  return total_loss