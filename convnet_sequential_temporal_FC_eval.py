import sys

from dataset import ImageSequenceDataset
from models.vgg_pretrained import SequentialImageTemporalFCModel
from train_evaluate_model import evaluate_model


SPATIAL_FULLY_CONNECTED = 256
TEMPORAL_FULLY_CONNECTED = [200]
BATCH_SIZE = 1
INPUT_SHAPE = [25, 140, 350, 3]


if __name__ == '__main__':
  dataset_root = sys.argv[1]
  model_path = sys.argv[2]

  dataset = ImageSequenceDataset(dataset_root, BATCH_SIZE, INPUT_SHAPE, is_training=False)

  model = SequentialImageTemporalFCModel(SPATIAL_FULLY_CONNECTED, TEMPORAL_FULLY_CONNECTED, dataset, is_training=False)

  evaluate_model(model, dataset, model_path)