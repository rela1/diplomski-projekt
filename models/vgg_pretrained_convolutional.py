import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from models.model_helper import read_vgg_init
from tensorflow.contrib.layers.python.layers import initializers

DROPOUT_KEEP_PROB = tf.constant(0.5, name='dropout_keep_prob')


class SequentialImageTemporalFCModelOnline:

  class SpatialsPart:

    def __init__(self, sequence_length, spatial_fully_connected_size, inputs, learning_rate, weight_decay=0.0, vgg_init_dir=None, is_training=False):
      self.build(sequence_length, spatial_fully_connected_size, inputs, learning_rate, weight_decay, vgg_init_dir, is_training)

    def build(self, sequence_length, spatial_fully_connected_size, inputs, learning_rate, weight_decay, vgg_init_dir, is_training):
      bn_params = {
      'decay': 0.999,
      'center': True,
      'scale': True,
      'epsilon': 0.001,
      'updates_collections': None,
      'is_training': is_training,
      }

      input_shape = inputs.get_shape()
      horizontal_slice_size = int(round(int(input_shape[2]) / 3))
      vertical_slice_size = int(round(int(input_shape[1]) / 3))
      inputs = tf.slice(inputs, begin=[0, vertical_slice_size, 0, 0], size=[-1, -1, horizontal_slice_size * 2, -1])

      self.final_gradient = tf.placeholder(tf.float32, shape=(1, spatial_fully_connected_size), name='_final_gradient_ph')
      self.handles = [None] * sequence_length

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
        net = layers.convolution2d(net, 512, scope='conv5_3', normalizer_fn=layers.batch_norm, normalizer_params=bn_params)
        net = layers.max_pool2d(net, 2, 2, scope='pool5')

      net = layers.flatten(net)

      with tf.contrib.framework.arg_scope([layers.fully_connected],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
        weights_initializer=layers.variance_scaling_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(net, spatial_fully_connected_size, scope='spatial_FC')

      self.representation = layers.flatten(net)

      self.loss = tf.matmul(self.representation, tf.transpose(self.final_gradient), name='_spatial_loss')

      self.partial_run_setup_objs = [self.representation, self.loss]
      if is_training:
        self.trainer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.trainer.minimize(self.loss)
        with tf.control_dependencies([self.train_op]):
          self.with_train_op = self.loss
        self.partial_run_setup_objs.append(self.train_op)

      vgg_layers, vgg_layer_names = read_vgg_init(vgg_init_dir)
      init_op, init_feed, pretrained_vars = create_init_op(vgg_layers)
      self.pretrained_vars = pretrained_vars
      self.vgg_init = (init_op, init_feed)


    def forward(self, sess, index):
      handle = sess.partial_run_setup([self.partial_run_setup_objs], [self.final_gradient])
      self.handles[index] = handle
      representation = sess.partial_run(self.handles[index], self.representation)
      return representation

    def backward(self, sess, final_gradient, index):
      loss, _ = sess.partial_run(self.handles[index], [self.loss, self.with_train_op], feed_dict={self.final_gradient: final_gradient})
      return loss


  class TemporalPart:

    def __init__(self, sequence_length, spatial_fully_connected_size, temporal_fully_connected_layers, labels, positive_class_weight, learning_rate, weight_decay=0.0, is_training=False):
      self.build(sequence_length, spatial_fully_connected_size, temporal_fully_connected_layers, labels, positive_class_weight, learning_rate, weight_decay=weight_decay, is_training=is_training)

    def build(self, sequence_length, spatial_fully_connected_size, temporal_fully_connected_layers, labels, positive_class_weight, learning_rate, weight_decay=0.0, is_training=False):
      bn_params = {
        'decay': 0.999,
        'center': True,
        'scale': True,
        'epsilon': 0.001,
        'updates_collections': None,
        'is_training': is_training,
      }
      self.sequence_new = tf.placeholder(tf.float32, shape=(1, spatial_fully_connected_size), name='_sequence_new_ph')
      self.sequence = tf.Variable(np.zeros((sequence_length, spatial_fully_connected_size)), dtype=tf.float32, trainable=False, name='_sequence_var')
      self.sequence_gradient = tf.Variable(np.zeros((sequence_length, spatial_fully_connected_size)), dtype=tf.float32, trainable=False, name='_sequence_grad')

      self.add_sequence_new_op = self.sequence.assign(tf.concat([tf.slice(self.sequence, begin=[1, 0], size=[-1, -1]), self.sequence_new], 0))
      self.add_sequence_gradient_new_op = self.sequence_gradient.assign(tf.concat([tf.slice(self.sequence_gradient, begin=[1, 0], size=[-1, -1]), tf.zeros_like(self.sequence_new)], 0))
      
      net = tf.reshape(self.sequence, (-1, sequence_length * spatial_fully_connected_size))

      with tf.control_dependencies([self.add_sequence_new_op, self.add_sequence_gradient_new_op]): 
        with tf.contrib.framework.arg_scope([layers.fully_connected],
          activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
          weights_initializer=layers.variance_scaling_initializer(),
          weights_regularizer=layers.l2_regularizer(weight_decay)):
          layer_num = 1
          for fully_connected_num in temporal_fully_connected_layers:
              net = layers.fully_connected(net, fully_connected_num, scope='temporal_FC{}'.format(layer_num))
              layer_num += 1

      self.logits = layers.fully_connected(
        net, 2, activation_fn=None, 
        weights_initializer=layers.xavier_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_initializer=tf.zeros_initializer(),
        scope='logits'
      )

      softmax = tf.nn.softmax(self.logits, name='_softmax')
      coefficients = tf.constant([1.0, positive_class_weight])
      cross_entropy = -tf.reduce_sum(tf.multiply(tf.one_hot(labels, depth=2) * tf.log(softmax + 1e-7), coefficients), reduction_indices=[1])
      xent_loss = tf.reduce_mean(cross_entropy, name='_temporal_loss')

      regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      self.loss = tf.add_n([xent_loss] + regularization_losses)
      self.labels = labels

      if is_training:
        self.trainer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.trainer.minimize(self.loss)
        self.sequence_gradient_new = tf.gradients(self.loss, [self.sequence])[0]
        self.add_sequence_gradient_new = tf.assign_add(self.sequence_gradient, self.sequence_gradient_new)

    def forward(self, sess, sequence_new,):
      logits, labels = sess.run([self.logits, self.labels], feed_dict={self.sequence_new: sequence_new})
      return logits, labels

    def forward_backward(self, sess, sequence_new):
      data = sess.run([self.loss, self.train_op, self.add_sequence_gradient_new, self.sequence_gradient_new, self.sequence, self.logits], feed_dict={self.sequence_new: sequence_new})
      return data

  def __init__(self, sequence_length, spatial_fully_connected_size, temporal_fully_connected_layers, dataset, learning_rate, weight_decay=0.0, vgg_init_dir=None, is_training=False):
    if is_training:
      with tf.variable_scope('model') as scope:
        self.spatials_train = self.SpatialsPart(sequence_length, spatial_fully_connected_size, dataset.train_images, learning_rate, weight_decay=weight_decay, vgg_init_dir=vgg_init_dir, is_training=True)
        self.vgg_init = self.spatials_train.vgg_init
        train_positive_class_weight = float(dataset.num_train_examples) / float(dataset.num_positive_train_examples)
        self.temporal_train = self.TemporalPart(sequence_length, spatial_fully_connected_size, temporal_fully_connected_layers, dataset.train_labels, train_positive_class_weight, learning_rate, weight_decay=weight_decay, is_training=True)
      with tf.variable_scope('model', reuse=True) as scope:
        self.spatials_valid = self.SpatialsPart(sequence_length, spatial_fully_connected_size, dataset.valid_images, learning_rate, weight_decay=weight_decay, vgg_init_dir=vgg_init_dir, is_training=False)
        valid_positive_class_weight = float(dataset.num_valid_examples) / float(dataset.num_positive_valid_examples)
        self.temporal_valid = self.TemporalPart(sequence_length, spatial_fully_connected_size, temporal_fully_connected_layers, dataset.valid_labels, valid_positive_class_weight, learning_rate, weight_decay=weight_decay, is_training=False)
        self.spatials_test = self.SpatialsPart(sequence_length, spatial_fully_connected_size, dataset.test_images, learning_rate, weight_decay=weight_decay, vgg_init_dir=vgg_init_dir, is_training=False)
        test_positive_class_weight = float(dataset.num_test_examples) / float(dataset.num_positive_test_examples)
        self.temporal_test = self.TemporalPart(sequence_length, spatial_fully_connected_size, temporal_fully_connected_layers, dataset.test_labels, test_positive_class_weight, learning_rate, weight_decay=weight_decay, is_training=False)
    else:
      with tf.variable_scope('model') as scope:
        self.spatials_train = self.SpatialsPart(sequence_length, spatial_fully_connected_size, dataset.train_images, learning_rate, weight_decay=weight_decay, vgg_init_dir=vgg_init_dir, is_training=False)
        train_positive_class_weight = float(dataset.num_train_examples) / float(dataset.num_positive_train_examples)
        self.temporal_train = self.TemporalPart(sequence_length, spatial_fully_connected_size, temporal_fully_connected_layers, dataset.train_labels, train_positive_class_weight, learning_rate, weight_decay=weight_decay, is_training=False)
      with tf.variable_scope('model', reuse=True) as scope:
        self.spatials_valid = self.SpatialsPart(sequence_length, spatial_fully_connected_size, dataset.valid_images, learning_rate, weight_decay=weight_decay, vgg_init_dir=vgg_init_dir, is_training=False)
        valid_positive_class_weight = float(dataset.num_valid_examples) / float(dataset.num_positive_valid_examples)
        self.temporal_valid = self.TemporalPart(sequence_length, spatial_fully_connected_size, temporal_fully_connected_layers, dataset.valid_labels, valid_positive_class_weight, learning_rate, weight_decay=weight_decay, is_training=False)
        self.spatials_test = self.SpatialsPart(sequence_length, spatial_fully_connected_size, dataset.test_images, learning_rate, weight_decay=weight_decay, vgg_init_dir=vgg_init_dir, is_training=False)
        test_positive_class_weight = float(dataset.num_test_examples) / float(dataset.num_positive_test_examples)
        self.temporal_test = self.TemporalPart(sequence_length, spatial_fully_connected_size, temporal_fully_connected_layers, dataset.test_labels, test_positive_class_weight, learning_rate, weight_decay=weight_decay, is_training=False)


def loss(logits, labels, is_training):
  unreduced_xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  xent_loss = tf.reduce_mean(unreduced_xent_loss)
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n([xent_loss] + regularization_losses, name='x___total_loss')
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