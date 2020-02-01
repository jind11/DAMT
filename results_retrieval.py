import os
import numpy as np
import sys

folder_name = sys.argv[1]
src = sys.argv[2]
tgt = sys.argv[3]

valid_str_name = '{}_{}-{}_mt_bleu'.format('valid', src, tgt)
test_str_name = '{}_{}-{}_mt_bleu'.format('test', src, tgt)
valid_scores = []
test_scores = []
with open(os.path.join(folder_name, 'train.log')) as ifile:
    for line in ifile:
        if valid_str_name in line and '__log__' not in line and 'INFO' in line and 'command' not in line \
                and 'best' not in line:
            score = float(line.strip().split(' -> ')[-1])
            valid_scores.append(score)
            continue
        if test_str_name in line and '__log__' not in line and 'INFO' in line and 'command' not in line \
                and 'best' not in line:
            score = float(line.strip().split(' -> ')[-1])
            test_scores.append(score)

assert len(valid_scores) == len(test_scores)

idx_max = np.argmax(valid_scores)
print('best valid score: ', max(valid_scores),
      'test score corresponding to best valid: ', test_scores[idx_max],
      'best test score: ', max(test_scores))