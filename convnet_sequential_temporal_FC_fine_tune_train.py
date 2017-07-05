import sys

from dataset import ImageSequenceDataset
from models.vgg_pretrained import SequentialImageTemporalFCModel
from train_evaluate_model import fine_tune_train_model


WEIGHT_DECAY = 5e-4
LEARNING_RATE = 1e-5
SPATIAL_FULLY_CONNECTED = 64
TEMPORAL_FULLY_CONNECTED = [64]
EPOCHS = 40
INPUT_SHAPE = [25, 140, 350, 3]


if __name__ == '__main__':
  vgg_init_dir = sys.argv[1]
  dataset_root = sys.argv[2]
  model_path = sys.argv[3]
  batch_size = int(sys.argv[4])

  dataset = ImageSequenceDataset(dataset_root, batch_size, INPUT_SHAPE, add_geolocations=True, is_training=True)

  model = SequentialImageTemporalFCModel(SPATIAL_FULLY_CONNECTED, TEMPORAL_FULLY_CONNECTED, dataset, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir, is_training=True)

  fine_tune_train_model(model, dataset, LEARNING_RATE, EPOCHS, model_path)