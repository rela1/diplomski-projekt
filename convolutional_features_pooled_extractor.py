import sys
import attributes_dataset as dataset
import tensorflow as tf
from models import vgg_vertically_sliced as model
import numpy as np
import os
import time

BATCH_SIZE = 5
WEIGHT_DECAY = 5e-4
DATASET_SPLITS = ['train', 'test', 'validate']

if __name__ == '__main__':
    dataset_folder = sys.argv[2]
    datasets = {}
    for dataset_split in DATASET_SPLITS:
        datasets[dataset_split] = dataset.read_images(dataset_folder, dataset_split)
        print("Read {} dataset...".format(dataset_split))
    for dataset_split in datasets:
        dataset = datasets[dataset_split]
        print("{} dataset shape: {}".format(dataset_split, dataset.shape))
    data_mean = datasets['train'].reshape([-1, 3]).mean(0)
    data_std = datasets['train'].reshape([-1, 3]).std(0)
    for c in range(datasets['train'].shape[-1]):
        datasets['train'][..., c] -= data_mean[c]
        datasets['test'][..., c] -= data_mean[c]
        datasets['validate'][..., c] -= data_mean[c]

        datasets['train'][..., c] /= data_std[c]
        datasets['test'][..., c] /= data_std[c]
        datasets['validate'][..., c] /= data_std[c]
    print("Normalized datasets...")
    sess = tf.Session()
    data_node = tf.placeholder(tf.float32, shape=(None, datasets['train'].shape[1], datasets['train'].shape[2], datasets['train'].shape[3]))
    with tf.variable_scope('model'):
        feature_extractor, init_op, init_feed = model.build_convolutional_pooled_feature_extractor(data_node, WEIGHT_DECAY, sys.argv[1])
    print("Created feature extractor...")
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    sess.run(init_op, feed_dict=init_feed)
    for dataset_name in datasets:
        current_dataset = datasets[dataset_name]
        dataset_features = []
        for i in range(len(current_dataset) // BATCH_SIZE + 1):
            start = time.clock()
            features = sess.run(feature_extractor, feed_dict={data_node:current_dataset[i * BATCH_SIZE : (i+1) * BATCH_SIZE]})
            dataset_features.extend(features)
            print("Done with feature extraction step, output shape: ", features.shape, " time per batch: ", (time.clock() - start))
        dataset_features = np.array(dataset_features)
        print("Done with feature extraction of {} dataset, final output shape: {}".format(dataset_name, dataset_features.shape))
        np.save(os.path.join(sys.argv[2], dataset_name + "_X_convolutional_pooled"), dataset_features)