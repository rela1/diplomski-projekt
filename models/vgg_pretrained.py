import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from models.model_helper import read_vgg_init
from tensorflow.contrib.layers.python.layers import initializers


class SequentialImageLSTMModel:

  def __init__(self, lstm_state_size, dataset, weight_decay=0.0, vgg_init_dir=None, is_training=False):
    if is_training:
      
      with tf.variable_scope('model'):
        logits, loss, init_op, init_feed = self.build(lstm_state_size, dataset.train_images, dataset.train_labels, weight_decay, vgg_init_dir, True)
        self.vgg_init = (init_op, init_feed)
        self.train_loss = loss
        self.train_logits = logits
      with tf.variable_scope('model', reuse=True):
        self.valid_logits, self.valid_loss = self.build(lstm_state_size, dataset.valid_images, dataset.valid_labels, weight_decay, vgg_init_dir, False)
        self.test_logits, self.test_loss = self.build(lstm_state_size, dataset.test_images, dataset.test_labels, weight_decay, vgg_init_dir, False)
    
    else:
      with tf.variable_scope('model'):
        self.train_logits, self.train_loss = self.build(lstm_state_size, dataset.train_images, dataset.train_labels, weight_decay, vgg_init_dir, False)
      with tf.variable_scope('model', reuse=True):
        self.valid_logits, self.valid_loss = self.build(lstm_state_size, dataset.valid_images, dataset.valid_labels, weight_decay, vgg_init_dir, False)
        self.test_logits, self.test_loss = self.build(lstm_state_size, dataset.test_images, dataset.test_labels, weight_decay, vgg_init_dir, False)

  def build(self, lstm_state_size, inputs, labels, weight_decay, vgg_init_dir, is_training):
    bn_params = {
      'decay': 0.999,
      'center': True,
      'scale': True,
      'epsilon': 0.001,
      'updates_collections': None,
      'is_training': is_training,
    }

    if is_training:
      vgg_layers, vgg_layer_names = read_vgg_init(vgg_init_dir)

    batch_size = tf.shape(inputs)[0]
    inputs_shape = inputs.get_shape()
    horizontal_slice_size = int(round(int(inputs_shape[3]) / 3))
    vertical_slice_size = int(round(int(inputs_shape[2]) / 3))
    inputs = tf.slice(inputs, begin=[0, 0, vertical_slice_size, 0, 0], size=[-1, -1, -1, horizontal_slice_size * 2, -1])

    concated = None

    sequence_length = int(inputs_shape[1])
    reuse = None
    for sequence_image in range(sequence_length):
      with tf.contrib.framework.arg_scope([layers.convolution2d],
        kernel_size=3, stride=1, padding='SAME', rate=1, activation_fn=tf.nn.relu,
        normalizer_fn=None, weights_initializer=None,
        weights_regularizer=layers.l2_regularizer(weight_decay)):

        net = layers.convolution2d(inputs[:, sequence_image], 64, scope='conv1_1', reuse=reuse)
        net = layers.convolution2d(net, 64, scope='conv1_2', reuse=reuse)
        net = layers.max_pool2d(net, 2, 2, scope='pool1')
        net = layers.convolution2d(net, 128, scope='conv2_1', reuse=reuse)
        net = layers.convolution2d(net, 128, scope='conv2_2', reuse=reuse)
        net = layers.max_pool2d(net, 2, 2, scope='pool2')
        net = layers.convolution2d(net, 256, scope='conv3_1', reuse=reuse)
        net = layers.convolution2d(net, 256, scope='conv3_2', reuse=reuse)
        net = layers.convolution2d(net, 256, scope='conv3_3', reuse=reuse)
        net = layers.max_pool2d(net, 2, 2, scope='pool3')
        net = layers.convolution2d(net, 512, scope='conv4_1', reuse=reuse)
        net = layers.convolution2d(net, 512, scope='conv4_2', reuse=reuse)
        net = layers.convolution2d(net, 512, scope='conv4_3', reuse=reuse)
        net = layers.max_pool2d(net, 2, 2, scope='pool4')
        net = layers.convolution2d(net, 512, scope='conv5_1', reuse=reuse)
        net = layers.convolution2d(net, 512, scope='conv5_2', reuse=reuse)
        net = layers.convolution2d(net, 512, scope='conv5_3', reuse=reuse)

        net = layers.batch_norm(net, decay=bn_params['decay'], center=bn_params['center'], 
                scale=bn_params['scale'], epsilon=bn_params['epsilon'], 
                updates_collections=bn_params['updates_collections'], is_training=bn_params['is_training'],
                scope='batch_norm', reuse=reuse)

      net_shape = net.get_shape()

      net = tf.reshape(net, [batch_size, int(net_shape[1]) * int(net_shape[2]) * int(net_shape[3])])

      if concated is None:
        concated = tf.expand_dims(net, axis=0)
      else:
        concated = tf.concat([concated, tf.expand_dims(net, axis=0)], axis=0)

    output_weights = tf.get_variable(
      'lstm_output_weights', 
      shape=[lstm_state_size, 2], 
      initializer=initializers.xavier_initializer(), 
      regularizer=layers.l2_regularizer(weight_decay)
    )
    output_bias = tf.get_variable(
      'lstm_output_bias', 
      shape=[2], 
      initializer=initializers.xavier_initializer(), 
      regularizer=layers.l2_regularizer(weight_decay)
    )
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_state_size)            

    if is_training:
      init_op, init_feed, pretrained_vars = create_init_op(vgg_layers)
      self.pretrained_vars = pretrained_vars

    net = tf.unstack(concated, num=sequence_length, axis=0)
    outputs, states = tf.contrib.rnn.static_rnn(lstm, net, dtype=tf.float32)
    
    logits = tf.matmul(outputs[-1], output_weights) + output_bias

    total_loss = loss(logits, labels, is_training)

    if is_training:
        return logits, total_loss, init_op, init_feed
    else:
        return logits, total_loss


class SequentialImageTemporalFCModel:

  def __init__(self, spatial_fully_connected_size, temporal_fully_connected_layers, dataset, weight_decay=0.0, vgg_init_dir=None, is_training=False):
    if is_training:
      
      with tf.variable_scope('model'):
        logits, loss, init_op, init_feed = self.build(spatial_fully_connected_size, temporal_fully_connected_layers, dataset.train_images, dataset.train_labels, weight_decay, vgg_init_dir, True)
        self.vgg_init = (init_op, init_feed)
        self.train_loss = loss
        self.train_logits = logits
      with tf.variable_scope('model', reuse=True):
        self.valid_logits, self.valid_loss = self.build(spatial_fully_connected_size, temporal_fully_connected_layers, dataset.valid_images, dataset.valid_labels, weight_decay, vgg_init_dir, False)
        self.test_logits, self.test_loss = self.build(spatial_fully_connected_size, temporal_fully_connected_layers, dataset.test_images, dataset.test_labels, weight_decay, vgg_init_dir, False)
    
    else:
      with tf.variable_scope('model'):
        self.train_logits, self.train_loss = self.build(spatial_fully_connected_size, temporal_fully_connected_layers, dataset.train_images, dataset.train_labels, weight_decay, vgg_init_dir, False)
      with tf.variable_scope('model', reuse=True):
        self.valid_logits, self.valid_loss = self.build(spatial_fully_connected_size, temporal_fully_connected_layers, dataset.valid_images, dataset.valid_labels, weight_decay, vgg_init_dir, False)
        self.test_logits, self.test_loss = self.build(spatial_fully_connected_size, temporal_fully_connected_layers, dataset.test_images, dataset.test_labels, weight_decay, vgg_init_dir, False)

  def build(self, spatial_fully_connected_size, temporal_fully_connected_layers, inputs, labels, weight_decay, vgg_init_dir, is_training):
    bn_params = {
      'decay': 0.999,
      'center': True,
      'scale': True,
      'epsilon': 0.001,
      'updates_collections': None,
      'is_training': is_training,
    }

    if is_training:
      vgg_layers, vgg_layer_names = read_vgg_init(vgg_init_dir)

    batch_size = tf.shape(inputs)[0]
    inputs_shape = inputs.get_shape()
    horizontal_slice_size = int(round(int(inputs_shape[3]) / 3))
    vertical_slice_size = int(round(int(inputs_shape[2]) / 3))
    inputs = tf.slice(inputs, begin=[0, 0, vertical_slice_size, 0, 0], size=[-1, -1, -1, horizontal_slice_size * 2, -1])


    concated = None

    reuse = None
    for sequence_image in range(int(inputs_shape[1])):
      with tf.contrib.framework.arg_scope([layers.convolution2d],
        kernel_size=3, stride=1, padding='SAME', rate=1, activation_fn=tf.nn.relu,
        normalizer_fn=None, weights_initializer=None,
        weights_regularizer=layers.l2_regularizer(weight_decay)):

        net = layers.convolution2d(inputs[:, sequence_image], 64, scope='conv1_1', reuse=reuse)
        net = layers.convolution2d(net, 64, scope='conv1_2', reuse=reuse)
        net = layers.max_pool2d(net, 2, 2, scope='pool1')
        net = layers.convolution2d(net, 128, scope='conv2_1', reuse=reuse)
        net = layers.convolution2d(net, 128, scope='conv2_2', reuse=reuse)
        net = layers.max_pool2d(net, 2, 2, scope='pool2')
        net = layers.convolution2d(net, 256, scope='conv3_1', reuse=reuse)
        net = layers.convolution2d(net, 256, scope='conv3_2', reuse=reuse)
        net = layers.convolution2d(net, 256, scope='conv3_3', reuse=reuse)
        net = layers.max_pool2d(net, 2, 2, scope='pool3')
        net = layers.convolution2d(net, 512, scope='conv4_1', reuse=reuse)
        net = layers.convolution2d(net, 512, scope='conv4_2', reuse=reuse)
        net = layers.convolution2d(net, 512, scope='conv4_3', reuse=reuse)
        net = layers.max_pool2d(net, 2, 2, scope='pool4')
        net = layers.convolution2d(net, 512, scope='conv5_1', reuse=reuse)
        net = layers.convolution2d(net, 512, scope='conv5_2', reuse=reuse)
        net = layers.convolution2d(net, 512, scope='conv5_3', reuse=reuse)

        net = layers.batch_norm(net, decay=bn_params['decay'], center=bn_params['center'], 
                scale=bn_params['scale'], epsilon=bn_params['epsilon'], 
                updates_collections=bn_params['updates_collections'], is_training=bn_params['is_training'],
                scope='batch_norm', reuse=reuse)

        net = layers.max_pool2d(net, 2, 2, scope='pool5')

      """
      net_shape = net.get_shape()

      global_pooling_kernel = [int(net_shape[1]), int(net_shape[2])]
      net = layers.max_pool2d(net, kernel_size=global_pooling_kernel, stride=global_pooling_kernel, scope='global_pool1')
      """
      net_shape = net.get_shape()

      net = tf.reshape(net, [batch_size, int(net_shape[1]) * int(net_shape[2]) * int(net_shape[3])])
        
      with tf.contrib.framework.arg_scope([layers.fully_connected],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(net, spatial_fully_connected_size, scope='spatial_FC', reuse=reuse)

      if concated is None:
        concated = tf.expand_dims(net, axis=1)
      else:
        concated = tf.concat([concated, tf.expand_dims(net, axis=1)], axis=1)

      reuse=True

    if is_training:
      init_op, init_feed, pretrained_vars = create_init_op(vgg_layers)
      self.pretrained_vars = pretrained_vars

    net = concated

    net_shape = net.get_shape()
    net = tf.reshape(net, [batch_size, int(net_shape[1]) * int(net_shape[2])])

    with tf.contrib.framework.arg_scope([layers.fully_connected],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay)):
        layer_num = 1
        for fully_connected_num in temporal_fully_connected_layers:
            net = layers.fully_connected(net, fully_connected_num, scope='temporal_FC{}'.format(layer_num))
            layer_num += 1

    logits = layers.fully_connected(net, 2, activation_fn=None, scope='logits')

    total_loss = loss(logits, labels, is_training)

    if is_training:
        return logits, total_loss, init_op, init_feed
    else:
        return logits, total_loss


class SequentialImagePoolingModel:

  def __init__(self, fully_connected_layers, dataset, weight_decay=0.0, vgg_init_dir=None, is_training=False):
    if is_training:
      
      with tf.variable_scope('model'):
        logits, loss, init_op, init_feed = self.build(fully_connected_layers, dataset.train_images, dataset.train_labels, weight_decay, vgg_init_dir, True)
        self.vgg_init = (init_op, init_feed)
        self.train_loss = loss
        self.train_logits = logits
      with tf.variable_scope('model', reuse=True):
        self.valid_logits, self.valid_loss = self.build(fully_connected_layers, dataset.valid_images, dataset.valid_labels, weight_decay, vgg_init_dir, False)
        self.test_logits, self.test_loss = self.build(fully_connected_layers, dataset.test_images, dataset.test_labels, weight_decay, vgg_init_dir, False)
    
    else:
      with tf.variable_scope('model'):
        self.train_logits, self.train_loss = self.build(fully_connected_layers, dataset.train_images, dataset.train_labels, weight_decay, vgg_init_dir, False)
      with tf.variable_scope('model', reuse=True):
        self.valid_logits, self.valid_loss = self.build(fully_connected_layers, dataset.valid_images, dataset.valid_labels, weight_decay, vgg_init_dir, False)
        self.test_logits, self.test_loss = self.build(fully_connected_layers, dataset.test_images, dataset.test_labels, weight_decay, vgg_init_dir, False)

  def build(self, fully_connected_layers, inputs, labels, weight_decay, vgg_init_dir, is_training):
    bn_params = {
      'decay': 0.999,
      'center': True,
      'scale': True,
      'epsilon': 0.001,
      'updates_collections': None,
      'is_training': is_training,
    }

    if is_training:
      vgg_layers, vgg_layer_names = read_vgg_init(vgg_init_dir)

    inputs_shape = inputs.get_shape()
    horizontal_slice_size = int(round(int(inputs_shape[3]) / 3))
    vertical_slice_size = int(round(int(inputs_shape[2]) / 3))
    inputs = tf.slice(inputs, begin=[0, 0, vertical_slice_size, 0, 0], size=[-1, -1, -1, horizontal_slice_size * 2, -1])

    reuse = None
    for sequence_image in range(int(inputs_shape[1])):
      with tf.contrib.framework.arg_scope([layers.convolution2d],
        kernel_size=3, stride=1, padding='SAME', rate=1, activation_fn=tf.nn.relu,
        normalizer_fn=None, weights_initializer=None,
        weights_regularizer=layers.l2_regularizer(weight_decay)):

        net = layers.convolution2d(inputs[:, sequence_image], 64, scope='conv1_1', reuse=reuse)
        net = layers.convolution2d(net, 64, scope='conv1_2', reuse=reuse)
        net = layers.max_pool2d(net, 2, 2, scope='pool1')
        net = layers.convolution2d(net, 128, scope='conv2_1', reuse=reuse)
        net = layers.convolution2d(net, 128, scope='conv2_2', reuse=reuse)
        net = layers.max_pool2d(net, 2, 2, scope='pool2')
        net = layers.convolution2d(net, 256, scope='conv3_1', reuse=reuse)
        net = layers.convolution2d(net, 256, scope='conv3_2', reuse=reuse)
        net = layers.convolution2d(net, 256, scope='conv3_3', reuse=reuse)
        net = layers.max_pool2d(net, 2, 2, scope='pool3')
        net = layers.convolution2d(net, 512, scope='conv4_1', reuse=reuse)
        net = layers.convolution2d(net, 512, scope='conv4_2', reuse=reuse)
        net = layers.convolution2d(net, 512, scope='conv4_3', reuse=reuse)
        net = layers.max_pool2d(net, 2, 2, scope='pool4')
        net = layers.convolution2d(net, 512, scope='conv5_1', reuse=reuse)
        net = layers.convolution2d(net, 512, scope='conv5_2', reuse=reuse)
        net = layers.convolution2d(net, 512, scope='conv5_3', reuse=reuse)

        net = layers.batch_norm(net, decay=bn_params['decay'], center=bn_params['center'], 
                scale=bn_params['scale'], epsilon=bn_params['epsilon'], 
                updates_collections=bn_params['updates_collections'], is_training=bn_params['is_training'],
                scope='batch_norm', reuse=reuse)

      if concated is None:
        concated = tf.expand_dims(net, axis=1)
      else:
        concated = tf.concat([concated, tf.expand_dims(net, axis=1)], axis=1)

      reuse=True

    if is_training:
      init_op, init_feed, pretrained_vars = create_init_op(vgg_layers)
      self.pretrained_vars = pretrained_vars

    net = concated

    net_shape = net.get_shape()
    batch_size = tf.shape(inputs)[0]

    net = tf.reshape(net, [batch_size, int(net_shape[1]), int(net_shape[2]) * int(net_shape[3]), int(net_shape[4])])
    net = layers.max_pool2d(net, kernel_size=2, stride=2, scope='pool5')

    net_shape = net.get_shape()
    net = tf.reshape(net, [batch_size, int(net_shape[1]) * int(net_shape[2]) * int(net_shape[3])])

    with tf.contrib.framework.arg_scope([layers.fully_connected],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay)):
        layer_num = 1
        for fully_connected_num in fully_connected_layers:
            net = layers.fully_connected(net, fully_connected_num, scope='fc{}'.format(layer_num))
            layer_num += 1

    logits = layers.fully_connected(net, 2, activation_fn=None, scope='logits')

    total_loss = loss(logits, labels, is_training)

    if is_training:
        return logits, total_loss, init_op, init_feed
    else:
        return logits, total_loss

class SingleImageModel:

  def __init__(self, fully_connected_layers, dataset, weight_decay=0.0, vgg_init_dir=None, is_training=False):
    if is_training:
      
      with tf.variable_scope('model'):
        logits, loss, init_op, init_feed = self.build(fully_connected_layers, dataset.train_images, dataset.train_labels, weight_decay, vgg_init_dir, True)
        self.vgg_init = (init_op, init_feed)
        self.train_loss = loss
        self.train_logits = logits
      with tf.variable_scope('model', reuse=True):
        self.valid_logits, self.valid_loss = self.build(fully_connected_layers, dataset.valid_images, dataset.valid_labels, weight_decay, vgg_init_dir, False)
        self.test_logits, self.test_loss = self.build(fully_connected_layers, dataset.test_images, dataset.test_labels, weight_decay, vgg_init_dir, False)
    
    else:
      with tf.variable_scope('model'):
        self.train_logits, self.train_loss = self.build(fully_connected_layers, dataset.train_images, dataset.train_labels, weight_decay, vgg_init_dir, False)
      with tf.variable_scope('model', reuse=True):
        self.valid_logits, self.valid_loss = self.build(fully_connected_layers, dataset.valid_images, dataset.valid_labels, weight_decay, vgg_init_dir, False)
        self.test_logits, self.test_loss = self.build(fully_connected_layers, dataset.test_images, dataset.test_labels, weight_decay, vgg_init_dir, False)

  def build(self, fully_connected_layers, inputs, labels, weight_decay, vgg_init_dir, is_training):
    bn_params = {
      'decay': 0.999,
      'center': True,
      'scale': True,
      'epsilon': 0.001,
      'updates_collections': None,
      'is_training': is_training,
    }

    if is_training:
      vgg_layers, vgg_layer_names = read_vgg_init(vgg_init_dir)

    inputs_shape = inputs.get_shape()
    horizontal_slice_size = int(round(int(inputs_shape[2]) / 3))
    vertical_slice_size = int(round(int(inputs_shape[1]) / 3))
    inputs = tf.slice(inputs, begin=[0, vertical_slice_size, 0, 0], size=[-1, -1, horizontal_slice_size * 2, -1])

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

    if is_training:
      init_op, init_feed, pretrained_vars = create_init_op(vgg_layers)
      self.pretrained_vars = pretrained_vars

    net = layers.batch_norm(net, decay=bn_params['decay'], center=bn_params['center'], 
      scale=bn_params['scale'], epsilon=bn_params['epsilon'], 
      updates_collections=bn_params['updates_collections'], is_training=bn_params['is_training'],
      scope='batch_norm'
    )

    net_shape = net.get_shape()
    batch_size = tf.shape(inputs)[0]
    net = tf.reshape(net, [batch_size, int(net_shape[1]) * int(net_shape[2]) * int(net_shape[3])])

    with tf.contrib.framework.arg_scope([layers.fully_connected],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_initializer=initializers.xavier_initializer(),
      weights_regularizer=layers.l2_regularizer(weight_decay)):
      layer_num = 1
      for fully_connected_num in fully_connected_layers:
        net = layers.fully_connected(net, fully_connected_num, scope='fc{}'.format(layer_num))
        layer_num += 1
        
    logits = layers.fully_connected(net, 2, activation_fn=None, scope='logits')

    total_loss = loss(logits, labels, is_training)

    if is_training:
      return logits, total_loss, init_op, init_feed

    else:
      return logits, total_loss


def loss(logits, labels, is_training):
  unreduced_xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  xent_loss = tf.reduce_mean(unreduced_xent_loss)
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n([xent_loss] + regularization_losses, name='total_loss')
  return total_loss


def create_init_op(vgg_layers):
  variables = tf.contrib.framework.get_variables()
  init_map = {}
  pretrained_vars = []
  for var in variables:
    name_split = var.name.split('/')
    if len(name_split) != 3:
      continue
    name = name_split[1] + '/' + name_split[2][:-2]
    if name in vgg_layers:
      print(var.name, ' --> init from ', name)
      init_map[var.name] = vgg_layers[name]
      pretrained_vars.append(var)
    else:
      print(var.name, ' --> random init')
  init_op, init_feed = tf.contrib.framework.assign_from_values(init_map)
  return init_op, init_feed, pretrained_vars