import os
import sys
import time
import matplotlib.image as img

import numpy as np
import tensorflow as tf

import attributes_dataset as dataset
from models import vgg_vertically_sliced
import evaluate_helper

np.set_printoptions(linewidth=250)

BATCH_SIZE = 10
FULLY_CONNECTED = [200]
NUM_CLASSES = 2

def evaluate(model, dataset_root, images_root, model_path, misclassified_output_folder):
  train_data, train_labels, validate_data, validate_labels, test_data, test_labels = dataset.read_and_normalize_images(dataset_root)

  with tf.Graph().as_default():
    sess = tf.Session()

    data_node = tf.placeholder(tf.float32,
        shape=(BATCH_SIZE, train_data.shape[1], train_data.shape[2], train_data.shape[3]))
    labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))

    with tf.variable_scope('model'):
      logits_eval, loss_eval = model.build(data_node, labels_node, NUM_CLASSES, fully_connected=FULLY_CONNECTED, is_training=False)

    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model'):
        print('name {}, shape {}'.format(var.name, var.get_shape()))   

    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    evaluate_helper.evaluate('train', train_data, train_labels, BATCH_SIZE, 
        evaluate_helper.tf_predict_func(sess, data_node, logits_eval), verbose=True)
    evaluate_helper.evaluate('validate', validate_data, validate_labels, BATCH_SIZE, 
        evaluate_helper.tf_predict_func(sess, data_node, logits_eval), verbose=True)
    test_metrics, test_labels_pred = evaluate_helper.evaluate('test', test_data, test_labels, BATCH_SIZE, 
        evaluate_helper.tf_predict_func(sess, data_node, logits_eval), verbose=True)

    test_images = dataset.read_images(images_root, 'test')

	for index, image in enumerate(test_images):
		if test_labels_pred[index] != test_labels[index]:
			img.imsave(os.path.join(misclassified_output_folder, str(test_labels[index]) + "_" + str(index) + ".png", image)			


if __name__ == '__main__':
  dataset_root = sys.argv[1]
  images_root = sys.argv[2]
  model_path = sys.argv[3]
  misclassified_output_folder = sys.argv[4]
  evaluate(vgg_vertically_sliced, dataset_root, images_root, model_path, misclassified_output_folder)