import sys

from dataset import ImageSequenceDataset
from models.vgg_pretrained import SequentialImageTemporalFCModel
from train_evaluate_model import plot_wrong_classifications


WEIGHT_DECAY = 5e-4
LSTM_STATE_SIZES = [128, 64, 32]
BATCH_SIZE = 2
INPUT_SHAPE = [25, 140, 350, 3]


if __name__ == '__main__':
  dataset_root = sys.argv[1]
  model_path = sys.argv[2]
  wrong_classified_save_path = sys.argv[3] if len(sys.argv) > 3 else None

  dataset = ImageSequenceDataset(dataset_root, BATCH_SIZE, INPUT_SHAPE, is_training=False)

  model = SequentialImageTemporalFCModel(SPATIAL_FULLY_CONNECTED, TEMPORAL_FULLY_CONNECTED, dataset, is_training=False)

  plot_wrong_classifications(model, dataset, model_path, wrong_classified_save_path)