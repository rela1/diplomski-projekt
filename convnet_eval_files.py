import os
import sys

if __name__ == '__main__':
    num_len = len(sys.argv[2])
    images = 0
    correct = 0
    fp = 0
    tp = 0
    tn = 0
    fn = 0
    for i in range(1, int(sys.argv[2]) + 1):
        file_path = str(i) + '.txt'
        true_file_path = 'true_' + ('0' * (num_len - len(str(i)))) + file_path
        true_file_path = os.path.join(sys.argv[1], true_file_path)
        pred_file_path = 'pred_' + ('0' * (num_len - len(str(i)))) + file_path
        pred_file_path = os.path.join(sys.argv[1], pred_file_path)
        with open(true_file_path) as f_true , open(pred_file_path) as f_pred:
            true = int(f_true.read())
            pred = int(f_pred.read())
            if true == 1:
                if pred == 1:
                    tp += 1
                    correct += 1
                else:
                    fn += 1
            else:
                if pred == 0:
                    tn += 1
                    correct += 1
                else:
                    fp += 1
        images += 1
    print('Accuracy {}'.format(correct / images))
    print('Recall {}'.format((tp) / (tp + fn)))
    print('Precision {}'.format((tp) / (tp + fp)))
    print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp, tn, fp, fn))