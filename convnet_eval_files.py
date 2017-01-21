import os
import sys

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        ranges = [(int(line.split('-')[0]), int(line.split('-')[1])) for line in f.read().splitlines()]
    num_len = len(sys.argv[2])
    images = 0
    correct = 0
    for i in range(1, int(sys.argv[2]) + 1):
        file_path = str(i) + '.txt'
        true_file_path = 'true_' + ('0' * (num_len - len(str(i)))) + file_path
        true_file_path = os.path.join(sys.argv[1], true_file_path)
        pred_file_path = 'pred_' + ('0' * (num_len - len(str(i)))) + file_path
        pred_file_path = os.path.join(sys.argv[1], pred_file_path)
        with open(true_file_path), open(pred_file_path) as f_true, f_pred:
            if f_true.read() == f_pred.read():
                correct += 1
        images += 1
    print('Accuracy {}'.format(correct / images))