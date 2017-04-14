import sys

from dataset import ImageSequenceDataset
from models.vgg_pretrained import SequentialImagePoolingModel
from train_evaluate_model import train_model


WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-4
FULLY_CONNECTED = [200]
EPOCHS = 50
BATCH_SIZE = 10
INPUT_SHAPE = [25, 40, 100, 3]


if __name__ == '__main__':
  vgg_init_dir = sys.argv[1]
  dataset_root = sys.argpv[2]
  model_path = sys.argv[3]

  dataset = ImageSequenceDataset(dataset_root, BATCH_SIZE, INPUT_SHAPE, is_training=True)

  model = SequentialImagePoolingModel(FULLY_CONNECTED, dataset, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir, is_training=True)

  train_model(model, dataset, LEARNING_RATE, EPOCHS, model_path)