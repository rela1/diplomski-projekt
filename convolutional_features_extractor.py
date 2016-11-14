import sys
import attributes_dataset as dataset
import tensorflow as tf
from models import vgg_vertically_sliced as model
import numpy as np
import os
import time

BATCH_SIZE = 5
WEIGHT_DECAY = 5e-4

if __name__ == '__main__':
	tf.app.flags.DEFINE_float('weight_decay', 5e-4, '')
	with tf.Graph().as_default():
		config = tf.ConfigProto(log_device_placement=False)
		sess = tf.Session(config=config)
		X = dataset.read_images(sys.argv[2], sys.argv[3])
		vertical_slice = int(sys.argv[4])
		vertical_slice_name = "left" if vertical_slice == 0 else "middle"
		data_node = tf.placeholder(tf.float32, shape=(None, int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])))
		with tf.variable_scope('model'):
			feature_extractor, init_op, init_feed = model.build_convolutional_feature_extractor(data_node, WEIGHT_DECAY, sys.argv[1], vertical_slice=vertical_slice)
			sess.run(tf.initialize_all_variables())
			sess.run(tf.initialize_local_variables())
			sess.run(init_op, feed_dict=init_feed)
			X_features = []
			for i in range(len(X) // BATCH_SIZE + 1):
				start = time.clock()
				x_features = sess.run(feature_extractor, feed_dict={data_node:X[i * BATCH_SIZE : (i+1) * BATCH_SIZE]})
				X_features.extend(x_features)
				print("Done with feature extraction step, output shape: ", x_features.shape, " time per batch: ", (time.clock() - start))
			X_features = np.array(X_features)
			print("Done with feature extraction, final output shape: ", X_features.shape)
			np.save(os.path.join(sys.argv[1], sys.argv[2]+"_X_convolutional_" + vertical_slice_name), X_features)