import matplotlib.pyplot as plt
import attributes_dataset as dataset
import sys
import numpy as np
import os

DATASET_SPLITS = ['test', 'validate', 'train']

if __name__ == '__main__':
    dataset_folder = sys.argv[1]
    for dataset_split in DATASET_SPLITS:
        images = dataset.read_images(dataset_folder, dataset_split)
        print("Read {} dataset...".format(dataset_split))
        print("{} dataset shape: {}".format(dataset_split, images.shape))
        if images.shape[3] == 4:
            print("Dataset has alpha channel - removing...")
            images = images[:, :, :, :3]
            print("Dataset alpha channel removed")
            np.save(os.path.join(dataset_folder, dataset_split + "_X"), images)
