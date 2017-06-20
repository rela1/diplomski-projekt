import sys

from dataset import ImageSequenceDataset
from models.vgg_pretrained import SequentialImageLSTMModel
from train_evaluate_model import fine_tune_train_model


WEIGHT_DECAY = 5e-4
LEARNING_RATE = 1e-5
LSTM_STATE_SIZES = [128, 64, 32]
EPOCHS = 40
BATCH_SIZE = 10
INPUT_SHAPE = [25, 140, 350, 3]


if __name__ == '__main__':
  vgg_init_dir = sys.argv[1]
  dataset_root = sys.argv[2]
  model_path = sys.argv[3]

  dataset = ImageSequenceDataset(dataset_root, BATCH_SIZE, INPUT_SHAPE, is_training=True)

  model = SequentialImageLSTMModel(LSTM_STATE_SIZES, dataset, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir, is_training=True)

  fine_tune_train_model(model, dataset, LEARNING_RATE, EPOCHS, model_path)