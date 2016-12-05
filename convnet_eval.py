import os
import sys
import time

import numpy as np
import tensorflow as tf

import attributes_dataset as dataset
from models import vgg_vertically_sliced
import evaluate_helper

np.set_printoptions(linewidth=250)

BATCH_SIZE = 10
FULLY_CONNECTED = [200]
NUM_CLASSES = 2

def evaluate(vgg_vertically_sliced, dataset_root, model_path):
  train_data, train_labels, validate_data, validate_labels, test_data, test_labels = dataset.read_and_normalize_images(dataset_root)

  with tf.Graph().as_default():
    sess = tf.Session()

    data_node = tf.placeholder(tf.float32,
        shape=(BATCH_SIZE, train_data.shape[1], train_data.shape[2], train_data.shape[3]))
    labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))

    with tf.variable_scope('model'):
      logits_eval, loss_eval = model.build(data_node, labels_node, NUM_CLASSES, fully_connected=FULLY_CONNECTED, is_training=False)

    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    sess.run(init_op, feed_dict=init_feed)

    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    evaluate_helper.evaluate('train', train_data, train_labels, BATCH_SIZE, 
        evaluate_helper.tf_predict_func(sess, inputs, logits_eval))
    evaluate_helper.evaluate('validate', validate_data, validate_labels, BATCH_SIZE, 
        evaluate_helper.tf_predict_func(sess, inputs, logits_eval))
    evaluate_helper.evaluate('test', test_data, test_labels, BATCH_SIZE, 
        evaluate_helper.tf_predict_func(sess, inputs, logits_eval))

if __name__ == '__main__':
  dataset_root = sys.argv[1]
  model_path = sys.argv[2]
  evaluate(vgg_vertically_sliced, dataset_root, model_path)
