import sys

from dataset import ImageSequenceDataset
from models.vgg_pretrained import SequentialImageLSTMModel
from train_evaluate_model import evaluate_model


WEIGHT_DECAY = 5e-3
LSTM_STATE_SIZES = [128, 64, 32]
BATCH_SIZE = 10
INPUT_SHAPE = [25, 140, 350, 3]


if __name__ == '__main__':
  dataset_root = sys.argv[1]
  model_path = sys.argv[2]

  dataset = ImageSequenceDataset(dataset_root, BATCH_SIZE, INPUT_SHAPE, is_training=False)

  model = SequentialImageLSTMModel(LSTM_STATE_SIZES, dataset, weight_decay=WEIGHT_DECAY, is_training=False)

  evaluate_model(model, dataset, model_path)