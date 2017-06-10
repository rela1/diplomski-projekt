import random
import sys
import os
import concurrent.futures
import multiprocessing

from skimage.exposure import equalize_adapthist
from matplotlib.image import imread, imsave
import numpy as np

from generate_dataset import write_example_sequence, get_images_sequence_and_single_image, load_intersection_lines, get_not_processed_video_names, download_video_geoinformation, index_video_geoinformation, download_video, get_video_info, get_positive_images_ranges, extract_video_frames, get_frames_resolution, create_tf_records_writer, write_processed_video_names, clear_redundant_data

SEQUENCE_HALF_LENGTH = 12
MIN_DISTANCE_TO_POSITIVE = 200 #meters
MIN_VIDEO_DURATION = 600 #seconds
SAME_TRESHOLD = 0.005
IMAGE_WIDTH = 350
IMAGE_HEIGHT = 140
MAX_DISTANCE_TO_INTERSECTION = 15 #meters
IMAGE_CHANNELS = 3


def extract_positive_examples(video_name, positive_images_ranges, frames_resolution, zero_pad_number, number_of_frames):
    os.mkdir(os.path.join(video_name, 'positives'))
    positive_examples = 0
    sequence_number = 1
    treshold = SAME_TRESHOLD * frames_resolution
    for positive_images_range in positive_images_ranges:
        prev_img = None
        sequence_dir = os.path.join(video_name, 'positives', str(sequence_number))
        os.mkdir(sequence_dir)
        image_number = 1
        image_zero_pad_number = len(str(positive_images_range[1] - positive_images_range[0] + 1))
        warmedup_sequence = False
        for positive_image in range(positive_images_range[0], positive_images_range[1] + 1):
            if not warmedup_sequence:
                single_img, single_img_eq, img_sequence_eq = get_images_sequence_and_single_image(positive_image, video_name, SEQUENCE_HALF_LENGTH * 2, 0, zero_pad_number, treshold, number_of_frames)
                if single_img is None:
                    continue
                for img_eq in img_sequence_eq:
                    imsave(os.path.join(sequence_dir, str(image_number).zfill(image_zero_pad_number) + '.png'), img_eq)
                    image_number += 1
                prev_img = single_img
                warmedup_sequence = True
                positive_examples += 1
            else:
                if positive_image >= 0 and positive_image <= number_of_frames:
                    img = imread(os.path.join(video_name, 'frames', str(positive_image).zfill(zero_pad_number) + '.png'))
                    if prev_img is not None:
                        diff = np.sum(np.abs(img - prev_img))
                        if diff < treshold:
                            continue
                    img_eq = equalize_adapthist(img, clip_limit=0.03)
                    imsave(os.path.join(sequence_dir, str(image_number).zfill(image_zero_pad_number) + '.png'), img_eq)
                    prev_img = img
                    positive_examples += 1
                    image_number += 1
        sequence_number += 1
    return positive_examples


def extract_negative_examples(video_name, number_of_positive_examples, speeds, positive_images_ranges, frames_resolution, sequential_tf_records_writer, zero_pad_number, number_of_frames, frames_per_second):
    treshold = SAME_TRESHOLD * frames_resolution
    average_speed = np.mean(speeds)
    min_frame_diff_to_positive = (MIN_DISTANCE_TO_POSITIVE / average_speed) * frames_per_second + 2 * SEQUENCE_HALF_LENGTH
    selected_single_images = set()
    while len(selected_single_images) < number_of_positive_examples:
        image = random.randint(1, number_of_frames)
        if image in selected_single_images:
            continue
        if any([abs(image - positive_images_range[0]) < min_frame_diff_to_positive or abs(image - positive_images_range[1]) < min_frame_diff_to_positive or (image >= positive_images_range[0] and image <= positive_images_range[1])
            for positive_images_range in positive_images_ranges]):
                continue
        single_img, single_img_eq, images_sequence_eq = get_images_sequence_and_single_image(image, video_name, SEQUENCE_HALF_LENGTH * 2, 0, zero_pad_number, treshold, number_of_frames)

        if single_img is not None:
            write_example_sequence(images_sequence_eq, 0, sequential_tf_records_writer)
            selected_single_images.add(image)


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
            frames_dir, number_of_frames, zero_pad_number = extract_video_frames(video_name, video_full_path, video_duration_seconds, frames_per_second, IMAGE_WIDTH, IMAGE_HEIGHT)
            frames_resolution = get_frames_resolution(frames_dir, zero_pad_number)

            sequential_tf_records_writer = create_tf_records_writer(os.path.join(video_name, video_name + '_sequential.tfrecords'))
            log_file.write('Created negatives sequential writer...\n')
            number_of_positive_examples = extract_positive_examples(video_name, positive_images_ranges, frames_resolution, zero_pad_number, number_of_frames)
            log_file.write('Extracted positive examples...\n')
            extract_negative_examples(video_name, number_of_positive_examples, speeds, positive_images_ranges, frames_resolution, sequential_tf_records_writer, zero_pad_number, number_of_frames, frames_per_second)
            log_file.write('Extracted negative examples...\n')
            sequential_tf_records_writer.close()
            log_file.write('Closed writer...\n')

            log_file.write('Number of positive examples: {}, number of negative examples: {}\n'.format(number_of_positive_examples, number_of_positive_examples))

            with open(os.path.join(video_name, 'examples.txt'), 'w') as examples_file:
                examples_file.write(str(number_of_positive_examples * 2))

    clear_redundant_data(found_intersections, frames_dir, log_file_path, video_full_path, video_name)

    return found_intersections


if __name__ == '__main__':
    
    intersection_lines_path = sys.argv[1]
    video_urls_path = sys.argv[2]
    downloaded_video_names_path = sys.argv[3]
    video_count = int(sys.argv[4])

    intersection_lines = load_intersection_lines(intersection_lines_path)

    not_processed_video_names, processed_video_names = get_not_processed_video_names(video_urls_path, downloaded_video_names_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_results = []
        for i in range(video_count):
            future_result = executor.submit(process_video, not_processed_video_names[i], intersection_lines, MAX_DISTANCE_TO_INTERSECTION)
            print('Submitted task of video {}'.format(not_processed_video_names[i]))
            future_results.append(future_result)
            processed_video_names.add(not_processed_video_names[i])
        for i in range(video_count):
            try:
                result = future_results[i].result()
                print('Done with video {} found intersections {}'.format(not_processed_video_names[i], result))
            except Exception as e:
                print('Exception during processing of video {}: \n{}'.format(not_processed_video_names[i], e))

    write_processed_video_names(processed_video_names, downloaded_video_names_path)