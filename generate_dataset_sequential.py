import sys
import subprocess
import urllib.request
import shutil
import tempfile
import json
import re
import os
from datetime import datetime
import math
import random

import tensorflow as tf
from matplotlib.image import imread
from skimage.exposure import equalize_adapthist
import numpy as np

from generate_dataset import load_intersection_lines, download_video, extract_video_frames, get_video_info, create_tf_records_writer, write_single_example, get_positive_images_ranges, index_video_geoinformation, download_video_geoinformation

IMAGE_WIDTH = 350
IMAGE_HEIGHT = 140
SAME_TRESHOLD = 0.005
MAX_DISTANCE_TO_INTERSECTION = 15 #meters
INFO_STEPS = 100


def extract_images(video_name, positive_images_ranges, tf_records_writer, zero_pad_number, treshold, number_of_frames):
    prev_img = None
    added_images = 0
    positive_examples = 0
    for i in range(1, number_of_frames + 1):
        if not i % INFO_STEPS:
            print('tfrecords step {}/{}'.format(i, number_of_frames))
        img = imread(os.path.join(video_name, 'frames', str(i).zfill(zero_pad_number) + '.png'))
        if prev_img is not None:
            diff = np.sum(np.abs(img - prev_img))
            if diff < treshold:
                continue
        prev_img = img
        img = equalize_adapthist(img, clip_limit=0.03)
        if any(lower <= i <= upper for (lower, upper) in positive_images_ranges):
            label = 1
            positive_examples += 1
        else:
            label = 0
        write_single_example(img.astype(dtype=np.float32), label, tf_records_writer)
        added_images += 1

    return added_images, positive_examples


if __name__ == '__main__':
    
    video_name = sys.argv[1]
    intersection_lines_path = sys.argv[2]

    json_geoinformation = download_video_geoinformation(video_name)

    points, times, speeds, time_offset, tree = index_video_geoinformation(json_geoinformation)

    intersection_lines = load_intersection_lines(intersection_lines_path)

    video_full_path = download_video(video_name)
    video_duration_seconds, frames_per_second = get_video_info(video_full_path)
    frames_dir, number_of_frames, zero_pad_number = extract_video_frames(video_name, video_full_path, video_duration_seconds, frames_per_second, IMAGE_WIDTH, IMAGE_HEIGHT)
    tf_records_writer = create_tf_records_writer(os.path.join(video_name, video_name + '_convolutional.tfrecords'))

    log_file_path = os.path.join(video_name, 'log.txt')

    with open(log_file_path, 'w') as log_file:
        positive_images_ranges = get_positive_images_ranges(intersection_lines, frames_per_second, tree, points, time_offset, times, speeds, MAX_DISTANCE_TO_INTERSECTION, log_file)
        
        added_images, positive_examples = extract_images(video_name, positive_images_ranges, tf_records_writer, zero_pad_number, IMAGE_WIDTH * IMAGE_HEIGHT * SAME_TRESHOLD, number_of_frames)   

        shutil.rmtree(frames_dir)
        os.remove(video_full_path)

        log_file.write('Number of positive examples: {}, number of negative examples: {}\n'.format(positive_examples, added_images - positive_examples))

        with open(os.path.join(video_name, 'examples.txt'), 'w') as examples_file:
                examples_file.write(str(added_images))