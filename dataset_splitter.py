import os
import sys

if __name__ == '__main__':
    root_dir = sys.argv[1]
    examples_per_video = {}
    for path in os.listdir(root_dir):
        full_path = os.path.join(root_dir, path)
        if os.path.isdir(full_path):
            log_file_path = os.path.join(full_path, 'log.txt')
            if os.path.isfile(log_file_path):
                with open(os.path.join) as f:
                    log = f.read()
                    number_of_positives = int(re.search('Number of positive examples: ([0-9]+?),', log).group(1))
                    number_of_examples = number_of_positives * 2
                    examples_per_video[path] = number_of_examples
    for video_name in sorted(examples_per_video, key=examples_per_video.get):
        print('{} -> {} examples'.format(video_name, examples_per_video[video_name]))