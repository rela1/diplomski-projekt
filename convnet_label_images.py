import math
import os
import sys
import time
import matplotlib.image as mpimg
from skimage import exposure

import numpy as np
import tensorflow as tf

import attributes_dataset as dataset
from models import vgg_vertically_sliced
import evaluate_helper

np.set_printoptions(linewidth=250)

BATCH_SIZE = 10
FULLY_CONNECTED = [200]
NUM_CLASSES = 2

def label(model, images_root_folder, model_path, model_input_size):

  with tf.Graph().as_default():
    sess = tf.Session()

    data_node = tf.placeholder(tf.float32,
        shape=(BATCH_SIZE, model_input_size[0], model_input_size[1], model_input_size[2]))
    labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))

    with tf.variable_scope('model'):
      logits_eval, loss_eval = model.build(data_node, labels_node, NUM_CLASSES, fully_connected=FULLY_CONNECTED, is_training=False)

    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model'):
        print('name {}, shape {}'.format(var.name, var.get_shape()))   

    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    image_paths = os.listdir(images_root_folder)
    image_paths = [os.path.join(images_root_folder, image_path) for image_path in image_paths]
    num_images = len(image_paths)
    num_batches = math.ceil(image_paths / BATCH_SIZE)
    for batch in range(num_batches):
      print('Batch {}/{}'.format(batch + 1, num_batches))
      batch_image_paths = image_paths[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
      batch_image_names = [os.path.basename(batch_image_path) for batch_image_path in batch_image_paths]
      batch_images = np.array([mpimg.imread(batch_image_path) for batch_image_path in batch_image_paths])
      batch_images = tf.image.resize_images(batch_images, model_input_size)
      batch_images = np.array([exposure.equalize_adapthist(batch_image, clip_limit=0.03) for batch_image in batch_images])
      batch_images_logits = sess.run(logits_eval, feed_dict={data_node : batch_images})
      batch_images_predicted = np.argmax(logits_eval, axis=1)
      for index, batch_image_path in enumerate(batch_image_paths):
        batch_image_name, batch_image_extension = os.path.splitext(batch_image_names[index])
        new_batch_image_path = os.path.join(images_root_folder, batch_image_name + '_' + str(batch_images_predicted[index]) + "." + batch_image_extension)
        print('Saved image {} as {}'.format(batch_image_paths, new_batch_image_path))
        #os.rename(batch_image_path, new_batch_image_path)


if __name__ == '__main__':
  images_root_folder = sys.argv[1]
  model_path = sys.argv[2]
  model_input_height = int(sys.argv[3])
  model_input_width = int(sys.argv[4])
  model_input_channels = int(sys.argv[5])
  label(vgg_vertically_sliced, images_root_folder, model_path, (model_input_height, model_input_width, model_input_channels))