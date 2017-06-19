import sys

from dataset import CombinedImageSequenceDataset
from models.vgg_pretrained import SequentialImageTemporalFCModelOnline, SequentialImageTemporalFCModel
from train_evaluate_model_combined import train_model


WEIGHT_DECAY = 5e-3
LEARNING_RATE = 5e-4
SPATIAL_FULLY_CONNECTED = 64
TEMPORAL_FULLY_CONNECTED = [64]
EPOCHS = 50
INPUT_SHAPE = [25, 140, 350, 3]
SEQUENCE_LENGTH = 25
BATCH_SIZE = 2


if __name__ == '__main__':
  vgg_init_dir = sys.argv[1]
  dataset_root = sys.argv[2]
  model_path = sys.argv[3]

  dataset = CombinedImageSequenceDataset(dataset_root, BATCH_SIZE, INPUT_SHAPE, is_training=True)

  fc_model = SequentialImageTemporalFCModel(SPATIAL_FULLY_CONNECTED, TEMPORAL_FULLY_CONNECTED, dataset, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir, is_training=True)
  convolutional_model = SequentialImageTemporalFCModelOnline(SEQUENCE_LENGTH, BATCH_SIZE, INPUT_SHAPE[1:], SPATIAL_FULLY_CONNECTED, TEMPORAL_FULLY_CONNECTED, LEARNING_RATE, weight_decay=WEIGHT_DECAY, is_training=True, reuse_weights=True)

  train_model(fc_model, convolutional_model, dataset, SEQUENCE_LENGTH, EPOCHS, LEARNING_RATE, model_path)