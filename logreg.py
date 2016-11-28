# coding=utf-8

import tensorflow as tf
import numpy as np

class TFLogReg:
  def __init__(self, D, C, param_delta=0.5, param_lambda=0):
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
       - param_delta: training step
    """
    with tf.variable_scope('model'):
      self.X = tf.placeholder(tf.float32, [None, D])
      self.Yoh_ = tf.placeholder(tf.float32, [None, C])

      self.W = tf.Variable(tf.random_normal([D, C], stddev=0.35))
      self.b = tf.Variable(tf.zeros(C))

      self.logits = tf.matmul(self.X, self.W) + self.b

      self.regularization_loss = tf.contrib.layers.l1_regularizer(param_lambda)(self.W)
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.Yoh_))
      self.total_loss = self.regularization_loss + self.loss

      self.trainer = tf.train.AdamOptimizer(learning_rate=param_delta)
      self.train_step = self.trainer.minimize(self.loss)

      self.session = tf.Session()

  def train(self, X, Yoh_, param_niter):
    """Arguments:
       - X: actual datapoints [NxD]
       - Yoh_: one-hot encoded labels [NxC]
       - param_niter: number of iterations
    """
    with tf.variable_scope('model'):
      self.session.run(tf.initialize_all_variables())
      for i in range(param_niter):
        operations = [self.train_step]
        if i % 10 == 0:
          operations.extend([self.total_loss, self.regularization_loss])
        session_results = self.session.run(operations, feed_dict={self.X : X, self.Yoh_ : Yoh_})
        if i % 10 == 0:
          print("iteration: {}, loss: {}, regularization loss: {}".format(i + 1, session_results[1], session_results[2]))

  def eval(self, X):
    """Arguments:
       - X: actual datapoints [NxD]
       Returns: predicted class probabilites [NxC]
    """
    with tf.variable_scope('model'):
      return self.session.run(self.logits, feed_dict={self.X : X})

  def attribute_value(self, attribute):
    with tf.variable_scope('model'):
      return self.session.run(attribute)