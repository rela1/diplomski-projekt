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

from generate_dataset import download_video, extract_video_frames, get_video_info, create_tf_records_writer, write_single_example

IMAGE_WIDTH = 700
IMAGE_HEIGHT = 280
SAME_TRESHOLD = 0.005
MIN_FRAME_DIFF_TO_POSITIVE = 150


def extract_positive_examples(video_name, positive_images_ranges, frames_resolution, tf_records_writer, zero_pad_number, number_of_frames):
    positive_examples = 0
    treshold = SAME_TRESHOLD * frames_resolution
    for positive_images_range in positive_images_ranges:
        prev_img = None
        for positive_image in range(positive_images_range[0], positive_images_range[1]):
            img = imread(os.path.join(video_name, 'frames', str(positive_image).zfill(zero_pad_number) + '.png'))
            if prev_img is not None:
                diff = np.sum(np.abs(img - prev_img))
                if diff < treshold:
                    continue
            prev_img = img
            img_eq = equalize_adapthist(img, clip_limit=0.03).astype(np.float32)
            write_single_example(img_eq, 1, tf_records_writer)
            positive_examples += 1
    return positive_examples


def extract_negative_examples(video_name, number_of_positive_examples, positive_images_ranges, frames_resolution, tf_records_writer, zero_pad_number, number_of_frames):
    selected_single_images = set()
    while len(selected_single_images) < number_of_positive_examples:
        image = random.randint(1, number_of_frames)
        if image in selected_single_images:
            continue
        if any([abs(image - positive_images_range[0]) < MIN_FRAME_DIFF_TO_POSITIVE or abs(image - positive_images_range[1]) < MIN_FRAME_DIFF_TO_POSITIVE or (image >= positive_images_range[0] and image <= positive_images_range[1])
            for positive_images_range in positive_images_ranges]):
                continue
        img = imread(os.path.join(video_name, 'frames', str(image).zfill(zero_pad_number) + '.png'))
        img_eq = equalize_adapthist(img, clip_limit=0.03).astype(np.float32)
        write_single_example(img_eq, 0, tf_records_writer)
        selected_single_images.add(image)


if __name__ == '__main__':
    
    video_name = sys.argv[1]
    positive_images_ranges_path = sys.argv[2]

    with open(positive_images_ranges_path) as f:
        positive_images_ranges = [[int(element) for element in line.split('-')] for line in f.read().splitlines()]

    video_full_path = download_video(video_name)
    video_duration_seconds, frames_per_second = get_video_info(video_full_path)
    frames_dir, number_of_frames, zero_pad_number = extract_video_frames(video_name, video_full_path, video_duration_seconds, frames_per_second, IMAGE_WIDTH, IMAGE_HEIGHT)
    tf_records_writer = create_tf_records_writer(os.path.join(video_name, video_name + '_single.tfrecords'))
    number_of_positive_examples = extract_positive_examples(video_name, positive_images_ranges, IMAGE_WIDTH * IMAGE_HEIGHT, tf_records_writer, zero_pad_number, number_of_frames)
    extract_negative_examples(video_name, number_of_positive_examples, positive_images_ranges, IMAGE_WIDTH * IMAGE_HEIGHT, tf_records_writer, zero_pad_number, number_of_frames)

    shutil.rmtree(frames_dir)
    os.remove(video_full_path)

    log_file_path = os.path.join(video_name, 'log.txt')

    with open(log_file_path, 'w') as log_file:
        log_file.write('Number of positive examples: {}, number of negative examples: {}\n'.format(number_of_positive_examples, number_of_positive_examples))