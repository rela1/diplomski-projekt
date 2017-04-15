import sys

from dataset import ImageSequenceDataset
from models.vgg_pretrained import SequentialImageTemporalFCModel
from train_evaluate_model import train_model


WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-4
SPATIAL_FULLY_CONNECTED = 64
TEMPORAL_FULLY_CONNECTED = [64]
EPOCHS = 50
BATCH_SIZE = 5
INPUT_SHAPE = [25, 40, 100, 3]


if __name__ == '__main__':
  vgg_init_dir = sys.argv[1]
  dataset_root = sys.argv[2]
  model_path = sys.argv[3]

  dataset = ImageSequenceDataset(dataset_root, BATCH_SIZE, INPUT_SHAPE, is_training=True)

  model = SequentialImageTemporalFCModel(SPATIAL_FULLY_CONNECTED, TEMPORAL_FULLY_CONNECTED, dataset, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir, is_training=True)

  train_model(model, dataset, LEARNING_RATE, EPOCHS, model_path)