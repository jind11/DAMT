import sys
import os
import random

data_name = sys.argv[1]
max_seq_len = int(sys.argv[2])

data_dir = '/data/medg/misc/jindi/nlp/datasets/OPUS/de-en'

# shuffle and then split the data into two halves to make sure the monolingual nature
src_data = open(os.path.join(data_dir, data_name, '{}-train.tok.de'.format(data_name.lower()))).readlines()
tgt_data = open(os.path.join(data_dir, data_name, '{}-train.tok.en'.format(data_name.lower()))).readlines()
assert len(src_data) == len(tgt_data)
perm = list(range(len(src_data)))
random.seed(1234)
random.shuffle(perm)
src_data = [src_data[idx] for idx in perm]
tgt_data = [tgt_data[idx] for idx in perm]
half_idx = len(src_data) // 2
src_data_trim = src_data[:half_idx]
tgt_data_trim = tgt_data[half_idx:]

open(os.path.join(data_dir, data_name, '{}-train.tok.de.mono'.format(data_name.lower())), 'w').write(''.join(src_data_trim))
open(os.path.join(data_dir, data_name, '{}-train.tok.en.mono'.format(data_name.lower())), 'w').write(''.join(tgt_data_trim))

# cut those sentences exceeding the max_seq_len in dev and test sets
for split in ['dev', 'test']:
    src_data = open(os.path.join(data_dir, data_name, '{}-{}.tok.de'.format(data_name.lower(), split))).readlines()
    tgt_data = open(os.path.join(data_dir, data_name, '{}-{}.tok.en'.format(data_name.lower(), split))).readlines()
    data_trim = [(line1, line2) for line1, line2 in zip(src_data, tgt_data) if len(line1.split()) <= max_seq_len and
                 len(line2.split()) <= max_seq_len]
    print('{} sentence exceeding length of {} in {}-{}.tok'.format(len(src_data)-len(data_trim),
                                                                      max_seq_len,
                                                                      data_name.lower(),
                                                                      split))
    open(os.path.join(data_dir, data_name, '{}-{}.tok.de.trim'.format(data_name.lower(), split)), 'w').write(
        ''.join(list(map(lambda x: x[0], data_trim))))
    open(os.path.join(data_dir, data_name, '{}-{}.tok.en.trim'.format(data_name.lower(), split)), 'w').write(
        ''.join(list(map(lambda x: x[1], data_trim))))