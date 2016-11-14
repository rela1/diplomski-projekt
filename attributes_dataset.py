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