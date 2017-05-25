import sys

from dataset import ConvolutionalImageData
from models.vgg_pretrained import SequentialImageTemporalFCModelOnline
from train_evaluate_model_online import train_model


WEIGHT_DECAY = 1e-3
LEARNING_RATE = 5e-4
SPATIAL_FULLY_CONNECTED = 64
TEMPORAL_FULLY_CONNECTED = [200]
EPOCHS = 100
INPUT_SHAPE = [140, 350, 3]
SEQUENCE_LENGTH = 25


if __name__ == '__main__':
  vgg_init_dir = sys.argv[1]
  dataset_root = sys.argv[2]
  model_path = sys.argv[3]

  dataset = ConvolutionalImageData(dataset_root, INPUT_SHAPE)

  model = SequentialImageTemporalFCModelOnline(SEQUENCE_LENGTH, SPATIAL_FULLY_CONNECTED, TEMPORAL_FULLY_CONNECTED, dataset, LEARNING_RATE, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir, is_training=True)

  train_model(model, dataset, SEQUENCE_LENGTH, EPOCHS, LEARNING_RATE, model_path)