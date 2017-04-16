import sys

from dataset import SingleImageDataset
from models.vgg_pretrained import SingleImageModel
from train_evaluate_model import evaluate_model


FULLY_CONNECTED = [200]
BATCH_SIZE = 10
INPUT_SHAPE = [280, 700, 3]


if __name__ == '__main__':
  dataset_root = sys.argv[1]
  model_path = sys.argv[2]

  dataset = SingleImageDataset(dataset_root, BATCH_SIZE, INPUT_SHAPE, is_training=False)

  model = SingleImageModel(FULLY_CONNECTED, dataset, is_training=False)

  evaluate_model(model, dataset, model_path)