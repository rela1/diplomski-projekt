import os
import sys

if __name__ == '__main__':
    with open(sys.argv[3], 'r') as f:
        ranges = [(int(line.split('-')[0]), int(line.split('-')[1])) for line in f.read().splitlines()]
    num_len = len(sys.argv[2])
    images = 0
    correct = 0
    fp = 0
    tp = 0
    tn = 0
    fn = 0
    fns = []
    fps = []
    for i in range(1, int(sys.argv[2]) + 1):
        file_path = str(i) + '.txt'
        true_file_path = 'true_' + ('0' * (num_len - len(str(i)))) + file_path
        true_file_path = os.path.join(sys.argv[1], true_file_path)
        pred_file_path = 'pred_new_' + ('0' * (num_len - len(str(i)))) + file_path
        pred_file_path = os.path.join(sys.argv[1], pred_file_path)
        with open(true_file_path) as f_true , open(pred_file_path) as f_pred:
            true = int(f_true.read())
            pred = int(f_pred.read())
            for r in ranges:
                if abs(r[0] - i) < 50  or abs(r[1] - i) < 50:
                    pred = true
                    break
            if true == 1:
                if pred == 1:
                    tp += 1
                    correct += 1
                else:
                    fn += 1
                    fns.append(i)
            else:
                if pred == 0:
                    tn += 1
                    correct += 1
                else:
                    fp += 1
                    fps.append(i)
        images += 1
    print('Accuracy {}'.format(correct / images))
    print('Recall {}'.format((tp) / (tp + fn)))
    print('Precision {}'.format((tp) / (tp + fp)))
    print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp, tn, fp, fn))
    print('FN indexes: {}'.format(fns))
    print('FP indexes: {}'.format(fps))