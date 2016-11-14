import os
import tensorflow as tf
import train_helper

MODEL_PATH = './models/vgg_vertically_sliced.py'
#MODEL_PATH = './models/swwae.py'

SAVE_DIR = os.path.join('/media/irelic/Storage/My Documents/Ivan/Škola/FER/9. semestar/Projekt/dataset/he_ftts-irap/merge_lane_images_dataset/results/', train_helper.get_time_string())

IMG_WIDTH = 492
IMG_HEIGHT = 147


tf.app.flags.DEFINE_string('optimizer', 'Adam', '')
# 1e-4 best, 1e-3 is too big
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4, '')
tf.app.flags.DEFINE_float('weight_decay', 5e-4, '')
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 3, '')
#tf.app.flags.DEFINE_integer('num_validations_per_epoch', 2, '')

#tf.app.flags.DEFINE_string('optimizer', 'Momentum', '')
##tf.app.flags.DEFINE_float('initial_learning_rate', 2e-4,
## 1e-3 the best
#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, """Initial learning rate.""")
##tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4, """Initial learning rate.""")
#tf.app.flags.DEFINE_float('momentum', 0.9, '')
##tf.app.flags.DEFINE_integer('num_epochs_per_decay', 3, '')
#tf.app.flags.DEFINE_integer('num_epochs_per_decay', 4, '')


tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
#tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.2,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, '')

tf.app.flags.DEFINE_string('vgg_init_dir', '/media/irelic/Storage/My Documents/Ivan/Škola/FER/9. semestar/Projekt/vgg16/', '')
#povecaj_lr za w=1
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', SAVE_DIR, \
    """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('resume_path', '', '')
tf.app.flags.DEFINE_integer('img_width', IMG_WIDTH, '')
tf.app.flags.DEFINE_integer('img_height', IMG_HEIGHT, '')
tf.app.flags.DEFINE_integer('num_channels', 3, '')

tf.app.flags.DEFINE_string('model_path', MODEL_PATH, '')
tf.app.flags.DEFINE_string('debug_dir', os.path.join(SAVE_DIR, 'debug'), '')

tf.app.flags.DEFINE_integer('max_epochs', 1, 'Number of epochs to run.')
#tf.app.flags.DEFINE_integer('batch_size', 50, '')
tf.app.flags.DEFINE_integer('batch_size', 10, '')
tf.app.flags.DEFINE_integer('num_classes', 2, '')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
tf.app.flags.DEFINE_boolean('draw_predictions', False, 'Whether to draw.')
tf.app.flags.DEFINE_boolean('save_net', True, 'Whether to save.')

tf.app.flags.DEFINE_integer('seed', 66478, '')

