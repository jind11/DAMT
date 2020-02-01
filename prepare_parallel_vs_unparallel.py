import sys
import os
import random

in_domain_data_name = sys.argv[1]
src_lang = sys.argv[2]
tgt_lang = sys.argv[3]
random_seed = int(sys.argv[4])
out_dir_para = sys.argv[5]
out_dir_unpara = sys.argv[6]

data_dir = '/data/medg/misc/jindi/nlp/datasets/OPUS/{}-{}'.format(src_lang, tgt_lang)
data_src = open(os.path.join(data_dir, in_domain_data_name,
                                       'train.{}'.format(src_lang))).readlines()
data_tgt = open(os.path.join(data_dir, in_domain_data_name,
                                       'train.{}'.format(tgt_lang))).readlines()

# sample in-domain data
random.seed(random_seed)
sample_idx = random.sample(range(len(data_src)), len(data_src)//2)
data_src_sample = [data_src[idx] for idx in sample_idx]
data_tgt_sample_para = [data_tgt[idx] for idx in sample_idx]
random.shuffle(data_tgt_sample_para)
data_tgt_sample_unpara = [data_tgt[idx] for idx in range(len(data_tgt)) if idx not in sample_idx]

os.makedirs(out_dir_para, exist_ok=True)
open(os.path.join(out_dir_para,
                  'train.{}'.format(src_lang)), 'w').write(''.join(data_src_sample))
open(os.path.join(out_dir_para,
                  'train.{}'.format(tgt_lang)), 'w').write(''.join(data_tgt_sample_para))

os.makedirs(out_dir_unpara, exist_ok=True)
open(os.path.join(out_dir_unpara,
                  'train.{}'.format(src_lang)), 'w').write(''.join(data_src_sample))
open(os.path.join(out_dir_unpara,
                  'train.{}'.format(tgt_lang)), 'w').write(''.join(data_tgt_sample_unpara))