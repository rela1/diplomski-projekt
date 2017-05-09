import sys

import tensorflow as tf

from dataset import SingleImageDataset
from models.vgg_pretrained import SingleImageModel
from train_evaluate_model import evaluate_model


FULLY_CONNECTED = [200]
BATCH_SIZE = 10
INPUT_SHAPE = [280, 700, 3]


if __name__ == '__main__':
  import pdb
  pdb.set_trace()
  dataset_root = sys.argv[1]
  model_path = sys.argv[2]
  resolution_factor = float(sys.argv[3])

  dataset = SingleImageDataset(dataset_root, BATCH_SIZE, INPUT_SHAPE, is_training=False)

  new_width = int(round(resolution_factor * INPUT_SHAPE[1]))
  new_height = int(round(resolution_factor * INPUT_SHAPE[0]))
  dataset.train_images = tf.image.resize_images(dataset.train_images, (new_height, new_width), tf.image.ResizeMethod.AREA)
  dataset.valid_images = tf.image.resize_images(dataset.valid_images, (new_height, new_width), tf.image.ResizeMethod.AREA)
  dataset.test_images = tf.image.resize_images(dataset.test_images, (new_height, new_width), tf.image.ResizeMethod.AREA)

  model = SingleImageModel(FULLY_CONNECTED, dataset, is_training=False)

  evaluate_model(model, dataset, model_path)