import sys
import os
import random

data_dir = sys.argv[1]
src_lg = sys.argv[2]
tgt_lg = sys.argv[3]
max_seq_len = int(sys.argv[4])

data_name = data_dir.split('/')[-1]
src_data = open(os.path.join(data_dir, '{}.{}-{}.{}'.format(data_name, src_lg, tgt_lg, src_lg)), 'r', encoding='utf-8-sig').readlines()
tgt_data = open(os.path.join(data_dir, '{}.{}-{}.{}'.format(data_name, src_lg, tgt_lg, tgt_lg)), 'r', encoding='utf-8-sig').readlines()

src_data_filter, tgt_data_filter = [], []
for (src_line, tgt_line) in zip(src_data, tgt_data):
    if len(src_line.split()) > max_seq_len or len(tgt_line.split()) > max_seq_len:
        continue
    if src_line.startswith('\ufeff') or tgt_line.startswith('\ufeff'):
        src_line = src_line.replace(u'\ufeff', '')
        tgt_line = tgt_line.replace(u'\ufeff', '').lstrip(' ')
        if not src_line.strip() or not tgt_line.strip():
            continue
    src_data_filter.append(src_line)
    tgt_data_filter.append(tgt_line)
print('number of filtered data: {}'.format(len(src_data_filter)))

src_data = src_data_filter
tgt_data = tgt_data_filter
assert len(src_data) == len(tgt_data)
perm_idx = list(range(len(src_data)))
random.seed(1234)
random.shuffle(perm_idx)

src_data = [src_data[idx] for idx in perm_idx]
tgt_data = [tgt_data[idx] for idx in perm_idx]

src_train = src_data[:-4000]
src_valid = src_data[-4000:-2000]
src_test = src_data[-2000:]

tgt_train = tgt_data[:-4000]
tgt_valid = tgt_data[-4000:-2000]
tgt_test = tgt_data[-2000:]

for split in ['train', 'valid', 'test']:
    for part in ['src', 'tgt']:
        open(os.path.join(data_dir, '{}.{}'.format(split, src_lg if part == 'src' else tgt_lg)), 'w').write(''.join(eval('{}_{}'.format(part, split))))

half_size = len(src_train) // 2
src_train_mono = src_train[:half_size]
tgt_train_mono = tgt_train[half_size:]
for part in ['src', 'tgt']:
    open(os.path.join(data_dir, 'train.{}.mono'.format(src_lg if part == 'src' else tgt_lg)), 'w').write(
        ''.join(eval('{}_train_mono'.format(part))))