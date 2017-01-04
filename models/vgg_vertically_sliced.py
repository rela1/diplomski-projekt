import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from models.model_helper import read_vgg_init
from tensorflow.contrib.layers.python.layers import initializers

def loss(logits, labels, is_training):
  xent_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  total_loss = tf.add_n([xent_loss] + regularization_losses, name='total_loss')
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

def build_convolutional_scaled_pooled_feature_extractor(inputs, scales=[1, 2], width_tiles=7, height_tiles=2, weight_decay=0.0, vgg_init_dir=None, is_training=True):
  if is_training:
    vgg_layers, vgg_layer_names = read_vgg_init(vgg_init_dir)

  output_maps = []

  for scale in scales:
    inputs_shape = inputs.get_shape()
    inputs_shape[1] = inputs_shape[1] / scale
    inputs_shape[2] = inputs_shape[2] / scale
    final_map_width = inputs_shape[2] // 32
    final_map_height = inputs_shape[1] // 32
    if final_map_height < height_tiles or final_map_width < width_tiles:
        raise Exception('Tiles (width, height) is :{}, final output map (width, height) on scale {} is: {}'.format((width_tiles, height_tiles), scale, (final_map_width, final_map_height)))

  with tf.variable_scope('operations', reuse=True):
    for scale in scales:
        inputs_shape = inputs.get_shape()
        if scale != 1:
            scaled_inputs = tf.image.resize_images(inputs, [inputs_shape[0], inputs_shape[1] / scale, inputs_shape[2] / scale, inputs_shape[3]])
        else:
            scaled_inputs = inputs
        scaled_inputs_shape = scaled_inputs.get_shape()
        horizontal_slice_size = int(round(int(scaled_inputs_shape[2]) / 3))
        vertical_slice_size = int(round(int(scaled_inputs_shape[1]) / 3))
        scaled_inputs = tf.slice(scaled_inputs, begin=[0, vertical_slice_size, 0, 0], size=[-1, vertical_slice_size, horizontal_slice_size * 2, -1])

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

            net_shape = net.get_shape()
            pool_height = int(round(net_shape[1] / height_tiles))
            pool_width = int(round(net_shape[2] / width_tiles))
            kernel = [pool_height, pool_width]
            net = layers.max_pool2d(net, kernel_size=kernel, stride=kernel)
            output_maps.append(net)
      
    packed_output_maps = tf.pack(output_maps)
    averaged_output_maps = tf.reduce_mean(packed_output_maps, reduction_indices=0)

    final_output = tf.contrib.layers.flatten(averaged_output_maps, scope='flatten')
    
    if is_training:
          init_op, init_feed = create_init_op(vgg_layers)
          return final_output, init_op, init_feed

    return final_output

def build_convolutional_pooled_feature_extractor(inputs, weight_decay=0.0, vgg_init_dir=None, is_training=True):
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
    net = layers.max_pool2d(net, 2, 2, scope='pool5', )

    #net = layers.convolution2d(net, 4096, kernel_size=7, scope='conv6_1')
    
    """
    last_convolution_filter_size = net.get_shape()
    horizontal_slice_size = int(round(int(last_convolution_filter_size[2]) / 3))
    vertical_slice_size = int(round(int(last_convolution_filter_size[1]) / 3))
    net = tf.slice(net, begin=[0, 0, 0, 0], size=[-1, vertical_slice_size * 2, horizontal_slice_size * 2, -1])
    print("Created slice from ", [0, 0, 0, 0], "with size ", [-1, vertical_slice_size * 2, horizontal_slice_size * 2, -1])
    print("Shape before pooling tiles: ", net.get_shape())
    """
    net = layers.max_pool2d(net, kernel_size=[2, 2], stride=2)
    print("Pooled tiles, new shape: ", net.get_shape())

    net = tf.contrib.layers.flatten(net, scope='flatten')

    if is_training:
      init_op, init_feed = create_init_op(vgg_layers)
      return net, init_op, init_feed

    return net

def build_convolutional_feature_extractor(inputs, weight_decay=0.0, vgg_init_dir=None, is_training=True):
  if is_training:
    vgg_layers, vgg_layer_names = read_vgg_init(vgg_init_dir)

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

    #net = layers.convolution2d(net, 4096, kernel_size=7, scope='conv6_1')
    
    last_convolution_filter_size = net.get_shape()
    net = layers.max_pool2d(net, kernel_size=[int(last_convolution_filter_size[1]), 1], stride=1)
    slice_size = int(round(int(last_convolution_filter_size[2]) / 3))
    net = tf.slice(net, begin=[0, 0, 0, 0], size=[-1, -1, slice_size * 2, -1])
    print("Created slice from ", [0, 0, 0, 0], "with size ", [-1, -1, slice_size * 2, -1])

    net = tf.contrib.layers.flatten(net, scope='flatten')

    if is_training:
      init_op, init_feed = create_init_op(vgg_layers)
      return net, init_op, init_feed

    return net

def build_scaled(inputs, labels, num_classes, scales=[1,2], width_tiles=7, height_tiles=2, fully_connected=[], weight_decay=0.0, vgg_init_dir=None, is_training=True):
  bn_params = {
      'decay': 0.999,
      'center': True,
      'scale': True,
      'epsilon': 0.001,
      'updates_collections': None,
      'is_training': is_training,
  }

  if is_training:
    net, init_op, init_feed = build_convolutional_scaled_pooled_feature_extractor(inputs, scales=scales, width_tiles=width_tiles, height_tiles=height_tiles, weight_decay=weight_decay, vgg_init_dir=vgg_init_dir, is_training=is_training)
  else:
    net = build_convolutional_scaled_pooled_feature_extractor(inputs, scales=scales, width_tiles=width_tiles, height_tiles=height_tiles, weight_decay=weight_decay, vgg_init_dir=vgg_init_dir, is_training=is_training)

  with tf.contrib.framework.arg_scope([layers.fully_connected],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_initializer=initializers.xavier_initializer(),
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    layer_num = 1
    for fully_connected_num in fully_connected:
      net = layers.fully_connected(net, fully_connected_num, scope='fc{}'.format(layer_num))
      layer_num += 1
  logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')

  total_loss = loss(logits, labels, is_training)
  
  if is_training:
    return logits, total_loss, init_op, init_feed

  return logits, total_loss   

def build(inputs, labels, num_classes, fully_connected=[], weight_decay=0.0, vgg_init_dir=None, is_training=True):
  bn_params = {
      'decay': 0.999,
      'center': True,
      'scale': True,
      'epsilon': 0.001,
      'updates_collections': None,
      'is_training': is_training,
  }

  if is_training:
    net, init_op, init_feed = build_convolutional_pooled_feature_extractor(inputs, weight_decay, vgg_init_dir, is_training)
  else:
    net = build_convolutional_pooled_feature_extractor(inputs, weight_decay, vgg_init_dir, is_training)

  with tf.contrib.framework.arg_scope([layers.fully_connected],
      activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_initializer=initializers.xavier_initializer(),
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    layer_num = 1
    for fully_connected_num in fully_connected:
      net = layers.fully_connected(net, fully_connected_num, scope='fc{}'.format(layer_num))
      layer_num += 1
  logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')

  total_loss = loss(logits, labels, is_training)
  
  if is_training:
    return logits, total_loss, init_op, init_feed

  return logits, total_loss