import sys

from dataset import ImageSequenceDataset
from models.vgg_pretrained import SequentialImagePoolingModel
from train_evaluate_model import freezed_pretrained_train_model


WEIGHT_DECAY = 5e-4
LEARNING_RATE = 1e-4
FULLY_CONNECTED = [200]
EPOCHS = 10
INPUT_SHAPE = [25, 40, 100, 3]


if __name__ == '__main__':
  vgg_init_dir = sys.argv[1]
  dataset_root = sys.argv[2]
  model_path = sys.argv[3]
  batch_size = int(sys.argv[4])

  dataset = ImageSequenceDataset(dataset_root, batch_size, INPUT_SHAPE, is_training=True)

  model = SequentialImagePoolingModel(FULLY_CONNECTED, dataset, weight_decay=WEIGHT_DECAY, vgg_init_dir=vgg_init_dir, is_training=True)

  freezed_pretrained_train_model(model, dataset, LEARNING_RATE, EPOCHS, model_path)