import sys

import tensorflow as tf

from dataset import SingleImageDataset
from models.vgg_pretrained import SingleImageModel
from train_evaluate_model import freezed_pretrained_train_model


WEIGHT_DECAY = 1e-3
LEARNING_RATE = 5e-4
FULLY_CONNECTED = [200]
EPOCHS = 10
BATCH_SIZE = 10
INPUT_SHAPE = [280, 700, 3]


if __name__ == '__main__':
  vgg_init_dir = sys.argv[1]
  dataset_root = sys.argv[2]
  model_path = sys.argv[3]
  resolution_factor = float(sys.argv[4])

  dataset = SingleImageDataset(dataset_root, BATCH_SIZE, INPUT_SHAPE, is_training=True)

  new_width = int(round(resolution_factor * INPUT_SHAPE[1]))
  new_height = int(round(resolution_factor * INPUT_SHAPE[0]))
  dataset.train_images = tf.image.resize_images(dataset.train_images, (new_height, new_width), tf.image.ResizeMethod.AREA)
  dataset.valid_images = tf.image.resize_images(dataset.valid_images, (new_height, new_width), tf.image.ResizeMethod.AREA)
  dataset.test_images = tf.image.resize_images(dataset.test_images, (new_height, new_width), tf.image.ResizeMethod.AREA)

  model = SingleImageModel(FULLY_CONNECTED, dataset, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir, is_training=True)

  freezed_pretrained_train_model(model, dataset, LEARNING_RATE, EPOCHS, model_path)