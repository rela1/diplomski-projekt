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

from fastkml import kml
from geopy.distance import vincenty
from sklearn.neighbors import KDTree
import tensorflow as tf
from matplotlib.image import imread
from skimage.exposure import equalize_adapthist
import numpy as np

MP4_VIDEO_FORMAT = 'https://he.ftts-irap.org/video/{}.mp4' 
JSON_GEOINFORMATION_FORMAT = 'https://he.ftts-irap.org/video/{}.json'
SEQUENCE_HALF_LENGTH = 12
MIN_DISTANCE_TO_POSITIVE = 100 #meters


def write_sequenced_and_single_example(single_image_frame, label, images_before_single, images_after_single, sequential_tf_records_writer, single_tf_records_writer, zero_pad_number, treshold, number_of_frames):
    images_sequence = []
    single_img = imread(os.path.join(video_name, 'frames', str(single_image_frame).zfill(zero_pad_number) + '.png'))

    i = 1
    prev_img = single_img
    added_images = 0
    while added_images < images_before_single:
        if single_image_frame - i <= 0:
            return False
        img = imread(os.path.join(video_name, 'frames', str(single_image_frame - i).zfill(zero_pad_number) + '.png'))
        diff = np.sum(np.abs(img - prev_img))
        i += 1
        if diff < treshold:
            continue
        else:
            prev_img = img
            img = equalize_adapthist(img, clip_limit=0.03)
            images_sequence.insert(0, img)
            added_images += 1

    single_img_eq = equalize_adapthist(single_img, clip_limit=0.03)
    images_sequence.append(single_img_eq)

    i = 1
    prev_img = single_img
    added_images = 0
    while added_images < images_after_single:
        if single_image_frame + i > number_of_frames:
            return False
        img = imread(os.path.join(video_name, 'frames', str(single_image_frame + i).zfill(zero_pad_number) + '.png'))
        diff = np.sum(np.abs(img - prev_img))
        i += 1
        if diff < treshold:
            continue
        else:
            prev_img = img
            img = equalize_adapthist(img, clip_limit=0.03)
            images_sequence.append(img)
            added_images += 1

    images_sequence = np.array(images_sequence)
    images_sequence_raw = images_sequence.tostring()

    sequence_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'height': _int64_feature(single_img.shape[0]),
                'width': _int64_feature(single_img.shape[1]),
                'depth': _int64_feature(single_img.shape[2]),
                'label': _int64_feature(label),
                'sequence_length': _int64_feature(images_sequence.shape[0]),
                'images_raw': _bytes_feature(images_sequence_raw)
            }
        )
    )
    sequential_tf_records_writer.write(sequence_example.SerializeToString())

    single_image_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'height': _int64_feature(single_img.shape[0]),
                'width': _int64_feature(single_img.shape[1]),
                'depth': _int64_feature(single_img.shape[2]),
                'label': _int64_feature(label),
                'image_raw': _bytes_feature(single_img_eq.tostring())
            }
        )
    )
    single_tf_records_writer.write(single_image_example.SerializeToString())

    return True


def get_frame_closest_to(point, frames_per_second, points_index_tree, points, time_offset, times, speeds, max_distance_to_intersection, log_file):
    dist, ind = tree.query([point], k=2)
    closer_point = points[ind[0][0]]
    further_point = points[ind[0][1]]
    closer_distance = vincenty(point, closer_point).meters
    further_distance = vincenty(point, further_point).meters
    if closer_distance < max_distance_to_intersection:
        closer_time = times[ind[0][0]]
        further_time = times[ind[0][1]]
        closer_speed = speeds[ind[0][0]]
        time_diff = closer_distance / closer_speed
        log_file.write('Closer time {}, closer distance {}, further time {}, further distance {}, closer speed {}, timediff {}\n'.format(closer_time, closer_distance, further_time, further_distance, closer_speed, time_diff))
        if closer_time < further_time:
            point_time = time_offset + closer_time + time_diff
        else:
            point_time = time_offset + closer_time - time_diff
        point_frame = round(frames_per_second * point_time)
        return point_time, point_frame
    else:
        return None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    
    video_name = sys.argv[1]

    if not os.path.exists(video_name):
        os.mkdir(video_name)

    video_full_path = os.path.join(video_name, video_name + '.mp4')

    with open(os.path.join(video_name, 'log.txt'), 'w') as log_file:

        # download video
        with urllib.request.urlopen(MP4_VIDEO_FORMAT.format(video_name)) as response, open(video_full_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

        # get video info
        video_info = subprocess.getoutput('ffprobe "{}"'.format(video_full_path))
        video_duration_string = re.search('Duration: (.*?),', video_info).group(1)
        frames_per_second = int(re.search('([0-9]+?) tbr', video_info).group(1))
        
        # extract video frames
        video_duration = datetime.strptime(video_duration_string, '%H:%M:%S.%f')
        video_duration_seconds = video_duration.hour * 3600 + video_duration.minute * 60 + video_duration.second + round(video_duration.microsecond / 1000000)
        approximate_number_of_frames = video_duration_seconds * frames_per_second
        frames_dir = os.path.join(video_name, 'frames')
        if not os.path.exists(frames_dir):
            os.mkdir(frames_dir)
        zero_pad_number = len(str(approximate_number_of_frames))
        frame_extract_info = subprocess.getoutput('ffmpeg -i "{}" -s {}x{} {}/frames/%0{}d.png'.format(video_full_path, sys.argv[3], sys.argv[4], video_name, zero_pad_number))
        number_of_frames = len(os.listdir(frames_dir))
        frames_resolution = int(sys.argv[3]) * int(sys.argv[4])

        # load intersection data
        k = kml.KML()
        with open(sys.argv[2], 'rb') as f:
            k.from_string(f.read())
        document = next(k.features())
        folder = next(document.features())
        intersection_lines = [placemark.geometry for placemark in folder.features()]

        # download video geoinformation
        with urllib.request.urlopen(JSON_GEOINFORMATION_FORMAT.format(video_name)) as response:
            json_geoinformation = json.loads(response.read().decode('utf-8'))

        # index geodata points
        points = []
        times = []
        speeds = []
        time_offset = json_geoinformation[0]['timeoffset']
        data_len = len(json_geoinformation)
        last_item_index = data_len - 1
        for i in range(1, data_len):
            points.append(json_geoinformation[i]['coordinates'])
            times.append(json_geoinformation[i]['time'])
            if i < last_item_index:
                distance = vincenty(json_geoinformation[i]['coordinates'], json_geoinformation[i + 1]['coordinates']).meters
                time = json_geoinformation[i + 1]['time'] - json_geoinformation[i]['time']
                speeds.append(distance / time)
            else:
                speeds.append(speeds[-1])
        tree = KDTree(points)

        # calculate positive image ranges
        positive_images_ranges = []
        max_distance_to_intersection = float(sys.argv[5])
        for intersection_line in intersection_lines:
            intersection_start = (intersection_line.coords[0][0], intersection_line.coords[0][1])
            intersection_end = (intersection_line.coords[-1][0], intersection_line.coords[-1][1])
            start_time_frame = get_frame_closest_to(intersection_start, frames_per_second, tree, points, time_offset, times, speeds, max_distance_to_intersection, log_file)
            end_time_frame = get_frame_closest_to(intersection_end, frames_per_second, tree, points, time_offset, times, speeds, max_distance_to_intersection, log_file)
            if start_time_frame is not None and end_time_frame is not None:
                if start_time_frame[0] < end_time_frame[0]:
                    positive_images_ranges.append((start_time_frame[1], end_time_frame[1]))
                else:
                    positive_images_ranges.append((end_time_frame[1], start_time_frame[1]))
        positive_images_ranges = sorted(positive_images_ranges)
        log_file.write('Positive images ranges {}\n'.format(positive_images_ranges))

        # create .tfrecords dataset files (sequence and middle files)
        sequential_tf_records_filename = os.path.join(video_name, video_name + '_sequential.tfrecords')
        middle_tf_records_filename = os.path.join(video_name, video_name + '_middle.tfrecords')

        sequential_tf_records_writer = tf.python_io.TFRecordWriter(
            sequential_tf_records_filename, 
            options=tf.python_io.TFRecordOptions(
                tf.python_io.TFRecordCompressionType.GZIP
            )
        )
        single_tf_records_writer = tf.python_io.TFRecordWriter(
            middle_tf_records_filename, 
            options=tf.python_io.TFRecordOptions(
                tf.python_io.TFRecordCompressionType.GZIP
            )
        )

        positive_examples = 0
        treshold = 0.005 * frames_resolution
        for positive_images_range in positive_images_ranges:
            prev_img = None
            for positive_image in range(positive_images_range[0], positive_images_range[1]):
                img = imread(os.path.join(video_name, 'frames', str(positive_image).zfill(zero_pad_number) + '.png'))
                if prev_img is not None:
                    diff = np.sum(np.abs(img - prev_img))
                    if diff < treshold:
                        continue
                prev_img = img
                if write_sequenced_and_single_example(positive_image, 1, SEQUENCE_HALF_LENGTH, SEQUENCE_HALF_LENGTH, sequential_tf_records_writer, single_tf_records_writer, zero_pad_number, treshold, number_of_frames):
                    positive_examples += 1
        log_file.write('Positive examples {}\n'.format(positive_examples))

        average_speed = np.mean(speeds)
        min_frame_diff_to_positive = (MIN_DISTANCE_TO_POSITIVE / average_speed) * frames_per_second + 2 * SEQUENCE_HALF_LENGTH
        selected_middle_images = set()
        while len(selected_middle_images) < positive_examples:
            image = random.randint(1, number_of_frames)
            if image in selected_middle_images:
                continue
            if any([abs(image - positive_images_range[0]) < min_frame_diff_to_positive or abs(image - positive_images_range[1]) < min_frame_diff_to_positive 
                for positive_images_range in positive_images_ranges]):
                    continue
            if write_sequenced_and_single_example(image, 0, SEQUENCE_HALF_LENGTH, SEQUENCE_HALF_LENGTH, sequential_tf_records_writer, single_tf_records_writer, zero_pad_number, treshold, number_of_frames):
                selected_middle_images.add(image)
        log_file.write('Number of examples {}\n'.format(positive_examples * 2))

        # delete frames and video file
        shutil.rmtree(frames_dir)
        os.remove(video_full_path)