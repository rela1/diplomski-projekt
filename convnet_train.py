import sys

from dataset import SingleImageDataset
from models.vgg_pretrained import SingleImageModel
from train_evaluate_model import train_model


WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-4
FULLY_CONNECTED = [200]
EPOCHS = 50
BATCH_SIZE = 10
INPUT_SHAPE = [250, 700, 3]


if __name__ == '__main__':
  vgg_init_dir = sys.argv[1]
  dataset_root = sys.argv[2]
  model_path = sys.argv[3]

  dataset = SingleImageDataset(dataset_root, BATCH_SIZE, INPUT_SHAPE, is_training=True)

  print('Dataset info: {} train examples, {} valid examples, {} test examples'.format(dataset.num_train_examples, dataset.num_valid_examples, dataset.num_test_examples))

  model = SingleImageModel(FULLY_CONNECTED, dataset, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir, is_training=True)

  train_model(model, dataset, LEARNING_RATE, EPOCHS, model_path)