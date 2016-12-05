import os, sys, tarfile, urllib
import numpy as np
import matplotlib.pyplot as plt

def read_labels(data_dir, split_name):
  """
  Args:
    split_name: name of the split (train/validate/test)
  Returns:
    label data array.
  """
  file_path = os.path.join(data_dir, split_name + '_y.npy')
  with open(file_path, 'rb') as f:
    labels = np.load(f)
  return np.ascontiguousarray(labels)

def read_convolutional_features(data_dir, vertical_slice, split_name):
  """
  Args:
    split_name: name of the split (train/validate/test)
  Returns:
    convolutional features data array.
  """
  file_path = os.path.join(data_dir, split_name + "_X_convolutional_" + vertical_slice + ".npy")
  with open(file_path, 'rb') as f:
    features = np.load(f)
  return np.ascontiguousarray(features)

def read_images(data_dir, split_name):
  """
  Args:
    split_name: name of the split (train/validate/test)
  Returns:
    image data array.
  """
  file_path = os.path.join(data_dir, split_name + '_X.npy')
  with open(file_path, 'rb') as f:
    images = np.load(f)
    images = np.transpose(images, (0, 1, 2, 3))
  return np.ascontiguousarray(images)

def read_and_normalize_images(data_dir):
  train_data = read_images(data_dir, 'train').astype(np.float64)
  test_data = read_images(data_dir, 'test').astype(np.float64)
  validate_data = read_images(data_dir, 'validate').astype(np.float64)
  train_labels = read_labels(data_dir, 'train').astype(np.int64)
  test_labels = read_labels(data_dir, 'test').astype(np.int64)
  validate_labels = read_labels(data_dir, 'validate').astype(np.int64)

  data_mean = train_data.reshape([-1, 3]).mean(0)
  data_std = train_data.reshape([-1, 3]).std(0)

  print('RGB mean = ', data_mean)
  print('RGB std = ', data_std)

  for c in range(train_data.shape[-1]):
    train_data[..., c] -= data_mean[c]
    test_data[..., c] -= data_mean[c]
    validate_data[..., c] -= data_mean[c]

    # better without variance normalization
    #train_data[..., c] /= data_std[c]
    #validate_data[..., c] /= data_std[c]
    #test_data[..., c] /= data_std[c]

  return train_data, train_labels, validate_data, validate_labels, test_data, test_labels