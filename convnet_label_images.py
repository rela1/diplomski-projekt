import glob
import math
import os
import sys
import time
import matplotlib.image as mpimg
from skimage import exposure
import scipy as sp

import numpy as np
import tensorflow as tf

import attributes_dataset as dataset
from models import vgg_vertically_sliced
import evaluate_helper

np.set_printoptions(linewidth=250)

BATCH_SIZE = 10
FULLY_CONNECTED = [200]
NUM_CLASSES = 2

def label(model, labels_root_folder, image_paths, model_path, model_input_size, rgb_mean):

  with tf.Graph().as_default():
    sess = tf.Session()

    data_node = tf.placeholder(tf.float32,
        shape=(None, model_input_size[0], model_input_size[1], model_input_size[2]))
    labels_node = tf.placeholder(tf.int64, shape=(None,))

    with tf.variable_scope('model'):
      logits_eval, loss_eval = model.build(data_node, labels_node, NUM_CLASSES, fully_connected=FULLY_CONNECTED, is_training=False)

    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model'):
        print('name {}, shape {}'.format(var.name, var.get_shape()))   

    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    num_images = len(image_paths)
    padd_len = len(str(num_images))
    num_batches = math.ceil(num_images / BATCH_SIZE)
    for batch in range(num_batches):
      print('Batch {}/{}'.format(batch + 1, num_batches))
      batch_image_paths = image_paths[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
      batch_image_names = [os.path.basename(batch_image_path) for batch_image_path in batch_image_paths]
      batch_images = np.array([exposure.equalize_adapthist(mpimg.imread(batch_image_path), clip_limit=0.03) for batch_image_path in batch_image_paths])
      for c in range(batch_images.shape[-1]):
      	batch_images[..., c] -= rgb_mean[c]
      batch_images_logits = sess.run(logits_eval, feed_dict={data_node : batch_images})
      batch_images_predicted = np.argmax(batch_images_logits, axis=1)
      for index, batch_image_path in enumerate(batch_image_paths):
        batch_image_name, batch_image_extension = os.path.splitext(batch_image_names[index])
        batch_label_path = batch_image_name + '.txt'
        batch_label_path = 'pred_new_' + ('0' * (padd_len - len(batch_image_name))) + batch_label_path
        batch_label_path = os.path.join(labels_root_folder, batch_label_path)
        with open(batch_label_path, 'w') as f:
          f.write(str(batch_images_predicted[index]))


if __name__ == '__main__':
  labels_root_folder = sys.argv[1]
  model_path = sys.argv[2]
  model_input_height = int(sys.argv[3])
  model_input_width = int(sys.argv[4])
  model_input_channels = int(sys.argv[5])
  rgb_mean = np.array(eval(sys.argv[6]))
  image_paths = glob.glob(sys.argv[7])
  label(vgg_vertically_sliced, labels_root_folder, image_paths, model_path, (model_input_height, model_input_width, model_input_channels), rgb_mean)