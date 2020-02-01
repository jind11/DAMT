import sys
import os
import random

in_domain_data_name = sys.argv[1]
src_lang = sys.argv[2]
tgt_lang = sys.argv[3]
sample_size = int(sys.argv[4])
out_dir = sys.argv[5]

data_dir = '/data/medg/misc/jindi/nlp/datasets/OPUS/{}-{}'.format(src_lang, tgt_lang)
in_domain_data_src = open(os.path.join(data_dir, in_domain_data_name, 'processed', '{}-{}'.format(src_lang, tgt_lang),
                                       'train.{}-{}.{}'.format(src_lang, tgt_lang, src_lang))).readlines()
in_domain_data_tgt = open(os.path.join(data_dir, in_domain_data_name, 'processed', '{}-{}'.format(src_lang, tgt_lang),
                                       'train.{}-{}.{}'.format(src_lang, tgt_lang, tgt_lang))).readlines()

# sample in-domain data
sample_idx = random.sample(range(len(in_domain_data_src)), sample_size)
in_domain_data_src_sample = [in_domain_data_src[idx] for idx in sample_idx]
in_domain_data_tgt_sample = [in_domain_data_tgt[idx] for idx in sample_idx]

os.makedirs(out_dir, exist_ok=True)
open(os.path.join(out_dir,
                  'train.{}-{}.{}'.format(src_lang, tgt_lang, src_lang)), 'w').write(''.join(in_domain_data_src_sample))
open(os.path.join(out_dir,
                  'train.{}-{}.{}'.format(src_lang, tgt_lang, tgt_lang)), 'w').write(''.join(in_domain_data_tgt_sample))