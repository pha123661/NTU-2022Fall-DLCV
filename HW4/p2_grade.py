import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('pred')
parser.add_argument('gt')
args = parser.parse_args()

with open(args.pred, 'r') as pred_f:
    with open(args.gt, 'r') as gt_f:
        p_reader = csv.reader(pred_f)
        gt_reader = csv.reader(gt_f)
        next(iter(p_reader))
        next(iter(gt_reader))

        correct_count = 0
        total_count = 0
        for (id_p, filename_p, pred), (id_gt, filename_gt, gt) in zip(p_reader, gt_reader):
            assert id_p == id_gt, (id_p, id_gt)
            assert filename_p == filename_gt, (filename_p, filename_gt)
            if pred == gt:
                correct_count += 1
            total_count += 1

print("ACC:", correct_count / total_count)
