import os
import sys

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        ranges = [(int(line.split('-')[0]), int(line.split('-')[1])) for line in f.read().splitlines()]
    num_len = len(sys.argv[2])
    for i in range(1, int(sys.argv[2]) + 1):
        file_path = str(i) + '.txt'
        file_path = 'true_' + '0' * (num_len - len(file_path) + 3) + file_path
        file_path = os.path.join(sys.argv[3], file_path)
        positive_image = False
        for r in ranges:
            if i >= r[0] and i <= r[1]:
                with open(file_path, 'w') as f:
                    f.write('1')
                positive_image = True
                break
        if not positive_image:
            with open(file_path, 'w') as f:
                f.write('0')
