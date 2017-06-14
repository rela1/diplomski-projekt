import sys

from dataset import ConvolutionalImageSequenceDataset
from models.vgg_pretrained_convolutional import SequentialImageTemporalFCModelOnline
from train_evaluate_model_convolutional import train_model


WEIGHT_DECAY = 5e-3
LEARNING_RATE = 5e-4
SPATIAL_FULLY_CONNECTED = 64
TEMPORAL_FULLY_CONNECTED = [64]
EPOCHS = 50
INPUT_SHAPE = [140, 350, 3]
SEQUENCE_LENGTH = 25
BATCH_SIZE = 10


if __name__ == '__main__':
  vgg_init_dir = sys.argv[1]
  dataset_root = sys.argv[2]
  model_path = sys.argv[3]

  dataset = ConvolutionalImageSequenceDataset(dataset_root, INPUT_SHAPE)

  convolutional_model = SequentialImageTemporalFCModelOnline(SEQUENCE_LENGTH, BATCH_SIZE, INPUT_SHAPE, SPATIAL_FULLY_CONNECTED, TEMPORAL_FULLY_CONNECTED, LEARNING_RATE, weight_decay=WEIGHT_DECAY, is_training=True)

  train_model(convolutional_model, dataset, SEQUENCE_LENGTH, EPOCHS, LEARNING_RATE, model_path, model_path)