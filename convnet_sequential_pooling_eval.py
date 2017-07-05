import sys

from dataset import ImageSequenceDataset
from models.vgg_pretrained import SequentialImagePoolingModel
from train_evaluate_model import evaluate_model


WEIGHT_DECAY = 5e-4
FULLY_CONNECTED = [200]
INPUT_SHAPE = [25, 140, 350, 3]
BATCH_SIZE = 2


if __name__ == '__main__':
  dataset_root = sys.argv[1]
  model_path = sys.argv[2]

  dataset = ImageSequenceDataset(dataset_root, BATCH_SIZE, INPUT_SHAPE, add_geolocations=True, is_training=False)

  model = SequentialImagePoolingModel(FULLY_CONNECTED, dataset, weight_decay=WEIGHT_DECAY, is_training=False)

  evaluate_model(model, dataset, model_path)