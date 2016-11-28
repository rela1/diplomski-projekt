import time
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import matplotlib.image as img
import sys
import attributes_dataset as dataset
import tensorflow as tf
import numpy as np
import os
from sklearn import metrics
from sklearn.metrics import confusion_matrix

EPOCHS = 100
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 20

def build_model(inputs, labels, weight_decay, hidden_layers, is_training=True):
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
  if is_training:
    session = tf.Session()
  net = inputs
  with tf.contrib.framework.arg_scope([layers.fully_connected], activation_fn=tf.nn.relu,
        weights_initializer=layers.xavier_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay), 
        normalizer_fn=layers.batch_norm, 
        normalizer_params=bn_params):
    for i in range(len(hidden_layers)):
      net = layers.fully_connected(net, hidden_layers[i], scope='fc{}'.format(i+1))
  logits = layers.fully_connected(net, 2, activation_fn=None, scope='logits')
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)) + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  if is_training:
    return session, logits, loss
  return logits, loss

def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def train(train_x, train_y, valid_x, valid_y, session, inputs, labels, logits, loss, starting_learning_rate):
  num_examples = train_x.shape[0]
  batch_size = BATCH_SIZE
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step,
                                           10000, 0.99, staircase=True)
  learning_step = (
    tf.train.AdamOptimizer(learning_rate)
    .minimize(loss, global_step=global_step)
  )
  session.run(tf.initialize_all_variables())
  max_epochs = EPOCHS
  best_valid_acc = 0
  saver = tf.train.Saver()
  for epoch in range(1, max_epochs+1):
    cnt_correct = 0
    train_x, train_y = shuffle_data(train_x, train_y)
    for i in range(num_batches):
      batch_x = train_x[i*batch_size:(i+1)*batch_size, :]
      batch_y = train_y[i*batch_size:(i+1)*batch_size, :]
      start_time = time.time()
      logits_val, loss_val, _ = session.run([logits, loss, learning_step], feed_dict={inputs:batch_x, labels:batch_y})
      duration = time.time() - start_time
      yp = np.argmax(logits_val, 1)
      yt = np.argmax(batch_y, 1)
      cnt_correct += (yp == yt).sum()
      if i % 5 == 0:
        sec_per_batch = float(duration)
        print("epoch %d, step %d/%d, batch loss = %.2f (%.3f sec/batch)" % (epoch, i*batch_size, num_examples, loss_val, sec_per_batch))
      if i > 0 and i % 50 == 0:
        print("Train accuracy = %.2f" % (cnt_correct / ((i+1)*batch_size) * 100))
    train_loss, train_acc = evaluate("Train", train_x, inputs, train_y, labels, session, logits, loss)     
    valid_loss, valid_acc = evaluate("Validation", valid_x, inputs, valid_y, labels, session, logits, loss)
    if valid_acc > best_valid_acc:
      best_valid_acc = valid_acc
      saved_path = saver.save(session, "best_model")
    saver.restore(session, saved_path)


def evaluate(name, x, inputs, y, labels, session, logits, loss):
  print("\nRunning evaluation: ", name)
  batch_size = BATCH_SIZE
  num_examples = x.shape[0]
  conf_matrix = np.zeros((2, 2))
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  loss_avg = 0
  for i in range(num_batches):
    batch_x = x[i*batch_size:(i+1)*batch_size, :]
    batch_y = y[i*batch_size:(i+1)*batch_size, :]
    logits_val, loss_val = session.run([logits, loss], feed_dict={inputs:batch_x, labels:batch_y})
    yp = np.argmax(logits_val, 1)
    yt = np.argmax(batch_y, 1)
    conf_matrix_batch = confusion_matrix(yt, yp, labels=np.arange(2))
    np.add(conf_matrix, conf_matrix_batch, conf_matrix)
    loss_avg += loss_val
  loss_avg /= num_batches
  total_conf_matrix_sum = np.sum(conf_matrix)
  row_conf_matrix_sum = np.sum(conf_matrix, axis = 1)
  column_conf_matrix_sum = np.sum(conf_matrix, axis = 0)
  diagonal_conf_matrix_sum = np.sum(np.diag(conf_matrix))
  acc = diagonal_conf_matrix_sum / total_conf_matrix_sum
  prec = [conf_matrix[i][i] / column_conf_matrix_sum[i] for i in range(2)]
  rec = [conf_matrix[i][i] / row_conf_matrix_sum[i] for i in range(2)]
  print(name + " accuracy = %.2f" % acc)
  print(name + " per class precision = %s" % prec)
  print(name + " per class recall = %s" % rec)
  print(name + " avg loss = %.2f\n" % loss_avg)
  return loss_avg, acc

if __name__ == '__main__':
  X_train_left = dataset.read_convolutional_features(sys.argv[1], sys.argv[2], 'train')
  X_train_middle = dataset.read_convolutional_features(sys.argv[1], sys.argv[3], 'train')
  X_train = np.concatenate((X_train_left, X_train_middle), axis=1)
  y_train = dataset.read_labels(sys.argv[1], 'train')
  y_train_oh = np.array([[1 if y_train[i] == j else 0 for j in range(2)] for i in range(len(y_train))])
  X_validate_left = dataset.read_convolutional_features(sys.argv[1], sys.argv[2], 'validate')
  X_validate_middle = dataset.read_convolutional_features(sys.argv[1], sys.argv[3], 'validate')
  X_validate = np.concatenate((X_validate_left, X_validate_middle), axis=1)
  y_validate = dataset.read_labels(sys.argv[1], 'validate')
  y_validate_oh = np.array([[1 if y_validate[i] == j else 0 for j in range(2)] for i in range(len(y_validate))])
  X_test_left = dataset.read_convolutional_features(sys.argv[1], sys.argv[2], 'test')
  X_test_middle = dataset.read_convolutional_features(sys.argv[1], sys.argv[3], 'test')
  X_test = np.concatenate((X_test_left, X_test_middle), axis=1)
  y_test = dataset.read_labels(sys.argv[1], 'test')
  y_test_oh = np.array([[1 if y_test[i] == j else 0 for j in range(2)] for i in range(len(y_test))])
  inputs = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, X_train.shape[1]))
  labels = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 2))
  with tf.variable_scope('model'):
    session, logits, loss = build_model(inputs, labels, WEIGHT_DECAY, [2000])
  with tf.variable_scope('model', reuse=True):
    logits_eval, loss_eval = build_model(inputs, labels, WEIGHT_DECAY, [2000], is_training=False)
  train(X_train, y_train_oh, X_validate, y_validate_oh, session, inputs, labels, logits, loss, LEARNING_RATE)
  evaluate("Test", X_test, inputs, y_test_oh, labels, session, logits_eval, loss_eval)
  if len(sys.argv) > 4:
    misclassified_output_folder = sys.argv[4]
    for index, image in enumerate(X_test_imgs):
      if y_test_pred[index] != y_test[index]:
        img.imsave(os.path.join(misclassified_output_folder, str(y_test[index]) + "_" + str(index)), image)