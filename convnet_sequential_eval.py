import sys

from dataset import ImageSequenceDataset
from models.vgg_pretrained import SequentialImagePoolingModel
from train_evaluate_model import evaluate_model


FULLY_CONNECTED = [200]
BATCH_SIZE = 5
INPUT_SHAPE = [25, 40, 100, 3]


if __name__ == '__main__':
  dataset_root = sys.argv[1]
  model_path = sys.argv[2]

  dataset = ImageSequenceDataset(dataset_root, BATCH_SIZE, INPUT_SHAPE, is_training=False)

  model = SequentialImagePoolingModel(FULLY_CONNECTED, dataset, is_training=False)

  evaluate(model, dataset, model_path)