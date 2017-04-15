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
MP4_VIDEO_REGEX = 'https://he.ftts-irap.org/video/(.*?).mp4'
JSON_GEOINFORMATION_FORMAT = 'https://he.ftts-irap.org/video/{}.json'
SEQUENCE_HALF_LENGTH = 12
MIN_DISTANCE_TO_POSITIVE = 200 #meters
MIN_VIDEO_DURATION = 600 #seconds
SAME_TRESHOLD = 0.005
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 40
SINGLE_IMAGE_WIDTH = 700
SINGLE_IMAGE_HEIGHT = 280
MAX_DISTANCE_TO_INTERSECTION = 15 #meters


class TFImageResize:

    def __init__(self):
        self.sess = tf.Session()

    def resize_images(self, images, dims):
        return self.sess.run(tf.image.resize_images(images, dims))


TF_IMAGE_RESIZER = TFImageResize()


def write_sequenced_and_single_example(single_image_frame, video_name, label, images_before_single, images_after_single, sequential_tf_records_writer, single_tf_records_writer, zero_pad_number, treshold, number_of_frames):
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

    images_sequence = np.array(images_sequence, dtype=np.float32)
    images_sequence_resized = TF_IMAGE_RESIZER.resize_images(images_sequence, (IMAGE_HEIGHT, IMAGE_WIDTH))
    images_sequence_raw = images_sequence_resized.astype(np.float32).tostring()

    sequence_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'height': _int64_feature(images_sequence_resized.shape[1]),
                'width': _int64_feature(images_sequence_resized.shape[2]),
                'depth': _int64_feature(images_sequence_resized.shape[3]),
                'label': _int64_feature(label),
                'sequence_length': _int64_feature(images_sequence_resized.shape[0]),
                'images_raw': _bytes_feature(images_sequence_raw)
            }
        )
    )
    sequential_tf_records_writer.write(sequence_example.SerializeToString())

    single_image_eq_resized = TF_IMAGE_RESIZER.resize_images(single_img_eq, (SINGLE_IMAGE_HEIGHT, SINGLE_IMAGE_WIDTH))
    print(single_image_eq_resized.shape, single_image_eq_resized.dtype)
    single_image_eq_raw = single_image_eq_resized.tostring()

    single_image_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'height': _int64_feature(single_image_eq_resized.shape[0]),
                'width': _int64_feature(single_image_eq_resized.shape[1]),
                'depth': _int64_feature(single_image_eq_resized.shape[2]),
                'label': _int64_feature(label),
                'image_raw': _bytes_feature(single_image_eq_raw)
            }
        )
    )
    single_tf_records_writer.write(single_image_example.SerializeToString())

    return True


def get_frame_closest_to(point, frames_per_second, points_index_tree, points, time_offset, times, speeds, max_distance_to_intersection, log_file):
    dist, ind = points_index_tree.query([point], k=2)
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


def download_video(video_name):
    if not os.path.exists(video_name):
        os.mkdir(video_name)

    video_full_path = os.path.join(video_name, video_name + '.mp4')

    if not os.path.exists(video_full_path):
        with urllib.request.urlopen(MP4_VIDEO_FORMAT.format(video_name)) as response, open(video_full_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

    return video_full_path


def get_video_info(video_full_path):
    video_info = subprocess.getoutput('ffprobe "{}"'.format(video_full_path))
    video_duration_string = re.search('Duration: (.*?),', video_info).group(1)
    frames_per_second = int(re.search('([0-9]+?) tbr', video_info).group(1))
    video_duration = datetime.strptime(video_duration_string, '%H:%M:%S.%f')
    video_duration_seconds = video_duration.hour * 3600 + video_duration.minute * 60 + video_duration.second + round(video_duration.microsecond / 1000000)
    return video_duration_seconds, frames_per_second


def download_video_geoinformation(video_name):
    with urllib.request.urlopen(JSON_GEOINFORMATION_FORMAT.format(video_name)) as response:
        json_geoinformation = json.loads(response.read().decode('utf-8'))
        return json_geoinformation


def index_video_geoinformation(json_geoinformation):
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

    return points, times, speeds, time_offset, tree


def load_intersection_lines(inersection_lines_path):
    k = kml.KML()
    with open(inersection_lines_path, 'rb') as f:
        k.from_string(f.read())
        document = next(k.features())
        folder = next(document.features())
        intersection_lines = [placemark.geometry for placemark in folder.features()]
        return intersection_lines


def get_positive_images_ranges(intersection_lines, frames_per_second, tree, points, time_offset, times, speeds, max_distance_to_intersection, log_file):
    positive_images_ranges = []
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
    return sorted(positive_images_ranges)


def extract_video_frames(video_name, video_full_path, video_duration_seconds, frames_per_second):
    approximate_number_of_frames = video_duration_seconds * frames_per_second
    zero_pad_number = len(str(approximate_number_of_frames))
    frames_dir = os.path.join(video_name, 'frames')

    if not os.path.exists(frames_dir):
        os.mkdir(frames_dir)
        frame_extract_info = subprocess.getoutput('ffmpeg -i "{}" -s {}x{} {}/frames/%0{}d.png'.format(video_full_path, SINGLE_IMAGE_WIDTH, SINGLE_IMAGE_HEIGHT, video_name, zero_pad_number))

    number_of_frames = len(os.listdir(frames_dir))
    return frames_dir, number_of_frames, zero_pad_number


def create_tf_records_writers(video_name):
    sequential_tf_records_filename = os.path.join(video_name, video_name + '_sequential.tfrecords')
    middle_tf_records_filename = os.path.join(video_name, video_name + '_single.tfrecords')

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

    return sequential_tf_records_writer, single_tf_records_writer


def extract_positive_examples(video_name, positive_images_ranges, frames_resolution, sequential_tf_records_writer, single_tf_records_writer, zero_pad_number, number_of_frames):
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
            if write_sequenced_and_single_example(positive_image, video_name, 1, SEQUENCE_HALF_LENGTH * 2, 0, sequential_tf_records_writer, single_tf_records_writer, zero_pad_number, treshold, number_of_frames):
                positive_examples += 1   
    return positive_examples


def extract_negative_examples(video_name, number_of_positive_examples, speeds, positive_images_ranges, frames_resolution, sequential_tf_records_writer, single_tf_records_writer, zero_pad_number, number_of_frames, frames_per_second):
    treshold = SAME_TRESHOLD * frames_resolution
    average_speed = np.mean(speeds)
    min_frame_diff_to_positive = (MIN_DISTANCE_TO_POSITIVE / average_speed) * frames_per_second + 2 * SEQUENCE_HALF_LENGTH
    selected_single_images = set()
    while len(selected_single_images) < number_of_positive_examples:
        image = random.randint(1, number_of_frames)
        if image in selected_single_images:
            continue
        if any([abs(image - positive_images_range[0]) < min_frame_diff_to_positive or abs(image - positive_images_range[1]) < min_frame_diff_to_positive 
            for positive_images_range in positive_images_ranges]):
                continue
        if write_sequenced_and_single_example(image, video_name, 0, SEQUENCE_HALF_LENGTH * 2, 0, sequential_tf_records_writer, single_tf_records_writer, zero_pad_number, treshold, number_of_frames):
            selected_single_images.add(image)


def clear_redundant_data(found_intersections, frames_dir, log_file_path, video_full_path, video_name):
    if not found_intersections:
        shutil.rmtree(video_name)
    else:
        shutil.rmtree(frames_dir)
        os.remove(video_full_path)


def get_frames_resolution(frames_dir, zero_pad_number):
    frame = imread(os.path.join(frames_dir, str(1).zfill(zero_pad_number) + '.png'))
    return frame.shape[0] * frame.shape[1]


def process_video(video_name, intersection_lines, max_distance_to_intersection):
    json_geoinformation = download_video_geoinformation(video_name)
    approximate_video_duration = json_geoinformation[-1]['time']

    if approximate_video_duration < MIN_VIDEO_DURATION:
        return

    points, times, speeds, time_offset, tree = index_video_geoinformation(json_geoinformation)
    video_full_path = download_video(video_name)
    video_duration_seconds, frames_per_second = get_video_info(video_full_path)
    log_file_path = os.path.join(video_name, 'log.txt')

    with open(log_file_path, 'w') as log_file:
        positive_images_ranges = get_positive_images_ranges(intersection_lines, frames_per_second, tree, points, time_offset, times, speeds, max_distance_to_intersection, log_file)
        found_intersections = len(positive_images_ranges) > 0

        frames_dir = None
        if found_intersections:
            log_file.write('Positive images ranges {}\n'.format(positive_images_ranges))
            frames_dir, number_of_frames, zero_pad_number = extract_video_frames(video_name, video_full_path, video_duration_seconds, frames_per_second)
            frames_resolution = get_frames_resolution(frames_dir, zero_pad_number)

            sequential_tf_records_writer, single_tf_records_writer = create_tf_records_writers(video_name)
            number_of_positive_examples = extract_positive_examples(video_name, positive_images_ranges, frames_resolution, sequential_tf_records_writer, single_tf_records_writer, zero_pad_number, number_of_frames)
            extract_negative_examples(video_name, number_of_positive_examples, speeds, positive_images_ranges, frames_resolution, sequential_tf_records_writer, single_tf_records_writer, zero_pad_number, number_of_frames, frames_per_second)

            log_file.write('Number of positive examples: {}, number of negative examples: {}\n'.format(number_of_positive_examples, number_of_positive_examples))

    clear_redundant_data(found_intersections, frames_dir, log_file_path, video_full_path, video_name)

    return found_intersections


def get_not_processed_video_names(video_urls_path, downloaded_video_names_path):
    
    with open(video_urls_path) as f:
        video_urls = f.read().splitlines()
    video_names = [re.search(MP4_VIDEO_REGEX, video_url).group(1) for video_url in video_urls]
    try:
        with open(downloaded_video_names_path, 'r') as f:
            downloaded_video_names = set(f.read().splitlines())
    except:
        downloaded_video_names = set()
    video_names = [video_name for video_name in video_names if not video_name in downloaded_video_names]
    return video_names, downloaded_video_names
    

def write_processed_video_names(processed_video_names, downloaded_video_names_path):
    with open(downloaded_video_names_path, 'w') as f:
        f.write('\n'.join(processed_video_names) + '\n')


if __name__ == '__main__':
    
    intersection_lines_path = sys.argv[1]
    video_urls_path = sys.argv[2]
    downloaded_video_names_path = sys.argv[3]
    video_count = int(sys.argv[4])

    intersection_lines = load_intersection_lines(intersection_lines_path)

    not_processed_video_names, processed_video_names = get_not_processed_video_names(video_urls_path, downloaded_video_names_path)

    processed_videos = 0
    for not_processed_video_name in not_processed_video_names:
        if process_video(not_processed_video_name, intersection_lines, MAX_DISTANCE_TO_INTERSECTION):
            processed_videos += 1
        processed_video_names.add(not_processed_video_name)
        if processed_videos >= video_count:
            break

    write_processed_video_names(processed_video_names, downloaded_video_names_path)