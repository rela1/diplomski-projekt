import os
import sys
import time
from shutil import copyfile

import numpy as np
import tensorflow as tf
#from tqdm import trange

import helper
import eval_helper
import train_helper
import attributes_dataset as dataset

np.set_printoptions(linewidth=250)

tf.app.flags.DEFINE_string('config_path', '', """Path to experiment config.""")
FLAGS = tf.app.flags.FLAGS

helper.import_module('config', FLAGS.config_path)
print(FLAGS.config_path)


def get_accuracy(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  predicted_labels = np.argmax(predictions, 1)
  assert predicted_labels.dtype == labels.dtype
  return 100.0 * np.sum(predicted_labels == labels) / predictions.shape[0]


def evaluate(sess, epoch_num, data_node, labels_node, logits, loss, data, labels):
  """ Trains the network
    Args:
      sess: TF session
      logits: network logits
  """
  print('\nTest performance:')
  loss_avg = 0
  batch_size = FLAGS.batch_size
  data_size = data.shape[0]
  print('testsize = ', data_size)
  assert data_size % batch_size == 0
  num_batches = data_size // batch_size
  correct_cnt = 0
  for step in range(num_batches):
    offset = step * batch_size 
    batch_data = data[offset:(offset + batch_size), ...]
    batch_labels = labels[offset:(offset + batch_size)]
    # This dictionary maps the batch data (as a numpy array) to the
    # node in the graph it should be fed to.
    feed_dict = {data_node: batch_data, labels_node: batch_labels}

    start_time = time.time()
    out_logits, loss_val = sess.run([logits, loss], feed_dict=feed_dict)
    duration = time.time() - start_time
    loss_avg += loss_val
    #net_labels = out_logits[0].argmax(2).astype(np.int32, copy=False)
    predicted_labels = out_logits.argmax(1)
    assert predicted_labels.dtype == batch_labels.dtype
    correct_cnt += np.sum(predicted_labels == batch_labels)

    if (step+1) % 10 == 0:
      num_examples_per_step = batch_size
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)
      format_str = 'epoch %d, step %d / %d, loss = %.2f \
                    (%.1f examples/sec; %.3f sec/batch)'
      print(format_str % (epoch_num, step+1, num_batches, loss_val,
                          examples_per_sec, sec_per_batch))
  print('')
  accuracy = 100 * correct_cnt / data_size
  print('Test accuracy = %.2f' % accuracy)
  return accuracy


def train(model):
  """ Trains the network
  Args:
    model: module containing model architecture
  """
  train_data = dataset.read_images('train').astype(np.float64)
  test_data = dataset.read_images('test').astype(np.float32)
  train_labels = dataset.read_labels('train').astype(np.int64)
  test_labels = dataset.read_labels('test').astype(np.int64)

  data_mean = train_data.reshape([-1, 3]).mean(0)
  data_std = train_data.reshape([-1, 3]).std(0)
  train_data = train_data.astype(np.float32)
  print('RGB mean = ', data_mean)
  print('RGB std = ', data_std)

  for c in range(train_data.shape[-1]):
    train_data[..., c] -= data_mean[c]
    test_data[..., c] -= data_mean[c]
    # better without variance normalization
    #train_data[..., c] /= data_std[c]
    #test_data[..., c] /= data_std[c]

  print(train_data.mean())
  print(train_data.std())
  print(train_data.flags['C_CONTIGUOUS'])
  print(train_labels.flags['C_CONTIGUOUS'])
  print(train_data.shape)
  print(train_labels.shape)

  train_size = train_data.shape[0]
  test_size = test_data.shape[0]
  batch_size = FLAGS.batch_size
  assert train_size % batch_size == 0
  assert test_size % batch_size == 0

  with tf.Graph().as_default():
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5 # don't hog all vRAM
    #config.operation_timeout_in_ms = 5000   # terminate on long hangs
    #config.operation_timeout_in_ms = 15000   # terminate on long hangs
    sess = tf.Session(config=config)
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable('global_step', [], dtype=tf.int64,
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = train_size // batch_size
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    data_node = tf.placeholder(tf.float32,
        shape=(batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.num_channels))
    labels_node = tf.placeholder(tf.int64, shape=(batch_size,))

    # Build a Graph that computes the logits predictions from the inference model.
    # Calculate loss.
    with tf.variable_scope('model'):
      #logits, loss, rec_img = model.build(data_node, labels_node)
      logits, loss, init_op, init_feed = model.build(data_node, labels_node, vertical_slice=0)
    with tf.variable_scope('model', reuse=True):
      logits_eval, loss_eval = model.build(data_node, labels_node, vertical_slice=0, is_training=False)
      #logits_eval, loss_eval, rec_img_eval = model.build(data_node, labels_node, is_training=False)
      #loss_valid = model.loss(logits_valid, labels_valid, weights_valid,
      #                        num_labels_valid, is_training=False)
    #logits_valid, loss_valid = logits, loss


    # Add a summary to track the learning rate.
    tf.scalar_summary('learning_rate', lr)
    #tf.scalar_summary('learning_rate', tf.mul(lr, tf.constant(1 / FLAGS.initial_learning_rate)))

    #with tf.control_dependencies([loss_averages_op]):
    opt = None
    if FLAGS.optimizer == 'Adam':
      opt = tf.train.AdamOptimizer(lr)
    elif FLAGS.optimizer == 'Momentum':
      opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum)
      #opt = tf.train.GradientDescentOptimizer(lr)
    else:
      raise ValueError()

    # Apply gradients.
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.histogram_summary(var.op.name, var)
    # Add histograms for gradients.
    grad_tensors = []
    for grad, var in grads:
      grad_tensors += [grad]
      #print(var)
      if grad is not None:
        tf.histogram_summary(var.op.name + '/gradients', grad)
    #grad = grads[-2][0]
    #print(grad)

    # Track the moving averages of all trainable variables.
    #variable_averages = tf.train.ExponentialMovingAverage(
    #    FLAGS.moving_average_decay, global_step)
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op]):
      train_op = tf.no_op(name='train')

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=FLAGS.max_epochs)
    #saver = tf.train.Saver(tf.all_variables())

    # Build an initialization operation to run below.
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    sess.run(init_op, feed_dict=init_feed)
    if len(FLAGS.resume_path) > 0:
      print('\nRestoring params from:', FLAGS.resume_path)
      assert tf.gfile.Exists(FLAGS.resume_path)
      #latest = tf.train.latest_checkpoint(FLAGS.train_dir)
      saver.restore(sess, FLAGS.resume_path)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)

    variable_map = train_helper.get_variable_map()
    # take the train loss moving average
    ex_start_time = time.time()
    num_batches = train_size // batch_size
    global_step_val = 0
    best_accuracy = 0
    accuracy_data = []

    visualize_dir = os.path.join(FLAGS.train_dir, 'visualize')
    for epoch_num in range(1, FLAGS.max_epochs + 1):
      print('\ntensorboard --logdir=' + FLAGS.train_dir + '\n')
      #conf_mat = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64)
      #conf_mat = np.ascontiguousarray(conf_mat)
      # Shuffle training data.
      indices = np.arange(train_size)
      np.random.shuffle(indices)
      train_data = np.ascontiguousarray(train_data[indices])
      #train_labels_old = train_labels
      train_labels = np.ascontiguousarray(train_labels[indices])
      #print((train_labels_old != train_labels).sum())
      #train_data = np.ascontiguousarray(train_data)
      #train_labels = np.ascontiguousarray(train_labels)

      for step in range(num_batches):
        offset = step * batch_size 
        batch_data = train_data[offset:(offset + batch_size), ...]
        batch_labels = train_labels[offset:(offset + batch_size)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {data_node: batch_data, labels_node: batch_labels}

        start_time = time.time()
        #run_ops = [train_op, loss, logits, rec_img, global_step]
        run_ops = [train_op, loss, logits, global_step]
        if global_step_val % 50 == 0:
          run_ops += [summary_op]
          ret_val = sess.run(run_ops, feed_dict=feed_dict)
          #(_, loss_val, scores, rec_img_val, global_step_val, summary_str) = ret_val
          (_, loss_val, scores, global_step_val, summary_str) = ret_val
          summary_writer.add_summary(summary_str, global_step_val)
        else:
          #run_ops += [grad_tensors]
          ret_val = sess.run(run_ops, feed_dict=feed_dict)
          #(_, loss_val, scores, rec_img_val, global_step_val) = ret_val
          (_, loss_val, scores, global_step_val) = ret_val
          #(_, loss_val, scores, yt, img_prefix, global_step_val, grads_val) = ret_val
          #train_helper.print_grad_stats(grads_val, grad_tensors)
        duration = time.time() - start_time
        #print(rec_img_val.min(), rec_img_val.max())
        #if rec_img is not None and global_step_val % 50 == 0:
        #  save_path = os.path.join(visualize_dir, str(global_step_val) + '_rec.png')
        #  train_helper.draw_tensor(rec_img_val[0], data_mean, data_std, save_path)
        #  save_path = os.path.join(visualize_dir, str(global_step_val) + '_input.png')
        #  train_helper.draw_tensor(batch_data[0], data_mean, data_std, save_path)

        #print(ret_val[5])
        #print('loss = ', ret_val[1])
        #print('logits min/max/mean = ', ret_val[2].min(), ret_val[2].max(), ret_val[2].mean())
        #print(ret_val[3].sum())

        assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

        # TODO estimate training accuracy on the last 30% of the epoch

        if (step+1) % 5 == 0:
          num_examples_per_step = batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = '%s: epoch %d, batch %d / %d, loss = %.2f \
            (%.1f examples/sec; %.3f sec/batch)'
          #print('lr = ', clr)
          print(format_str % (train_helper.get_expired_time(ex_start_time), epoch_num,
                              step+1, num_batches, loss_val, examples_per_sec, sec_per_batch))
          #print('Minibatch accuracy = ', get_accuracy(scores, batch_labels))
      accuracy = evaluate(sess, epoch_num, data_node, labels_node, logits_eval,
                          loss_eval, test_data, test_labels)
      accuracy_data += [accuracy]
      if accuracy > best_accuracy:
        best_accuracy = accuracy
      print('Best test accuracy = %.2f' % best_accuracy)


def main(argv=None):  # pylint: disable=unused-argument
  model = helper.import_module('model', FLAGS.model_path)

  if tf.gfile.Exists(FLAGS.train_dir):
    raise ValueError('Train dir exists: ' + FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  stats_dir = os.path.join(FLAGS.train_dir, 'stats')
  tf.gfile.MakeDirs(stats_dir)
  tf.gfile.MakeDirs(os.path.join(FLAGS.train_dir, 'visualize'))
  f = open(os.path.join(stats_dir, 'log.txt'), 'w')
  sys.stdout = train_helper.Logger(sys.stdout, f)

  copyfile(FLAGS.model_path, os.path.join(FLAGS.train_dir, 'model.py'))
  copyfile(FLAGS.config_path, os.path.join(FLAGS.train_dir, 'config.py'))

  print('Experiment dir: ' + FLAGS.train_dir)
  train(model)


if __name__ == '__main__':
  tf.app.run()

