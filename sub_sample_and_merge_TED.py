import sys
import os
import random

in_domain_data_name = sys.argv[1]
out_domain_data_name = sys.argv[2]
src_lang = sys.argv[3]
tgt_lang = sys.argv[4]
sample_size = int(sys.argv[5])
out_dir = sys.argv[6]

data_dir = '/data/medg/misc/jindi/nlp/datasets/OPUS/{}-{}'.format(src_lang, tgt_lang)
in_domain_data_src = open(os.path.join(data_dir, in_domain_data_name, 'processed', '{}-{}'.format(src_lang, tgt_lang),
                                       'train.{}'.format(src_lang))).readlines()
in_domain_data_tgt = open(os.path.join(data_dir, in_domain_data_name, 'processed', '{}-{}'.format(src_lang, tgt_lang),
                                       'train.{}'.format(tgt_lang))).readlines()

out_domain_data_src = open(os.path.join(data_dir, out_domain_data_name, 'processed', '{}-{}'.format(src_lang, tgt_lang),
                                       'train.{}'.format(src_lang))).readlines()
out_domain_data_tgt = open(os.path.join(data_dir, out_domain_data_name, 'processed', '{}-{}'.format(src_lang, tgt_lang),
                                       'train.{}'.format(tgt_lang))).readlines()

# sample in-domain data
sample_size = min(sample_size, len(in_domain_data_src))
sample_idx = random.sample(range(len(in_domain_data_src)), sample_size)
in_domain_data_src_sample = [in_domain_data_src[idx] for idx in sample_idx]
in_domain_data_tgt_sample = [in_domain_data_tgt[idx] for idx in sample_idx]

# over-sampling
ratio = len(out_domain_data_src) * 1. / len(in_domain_data_src_sample)
if ratio > 1:
    in_domain_data_src_final = int(ratio) * in_domain_data_src_sample
    in_domain_data_tgt_final = int(ratio) * in_domain_data_tgt_sample
    left_sample_idx = random.sample(range(len(in_domain_data_src_sample)), len(out_domain_data_src)
                                    - len(in_domain_data_src_final))
    in_domain_data_src_final += [in_domain_data_src_sample[idx] for idx in left_sample_idx]
    in_domain_data_tgt_final += [in_domain_data_tgt_sample[idx] for idx in left_sample_idx]
    assert len(out_domain_data_src) == len(in_domain_data_src_final)
    # assert len(out_domain_data_tgt) == len(in_domain_data_tgt_final)
else:
    in_domain_data_src_final = in_domain_data_src_sample
    in_domain_data_tgt_final = in_domain_data_tgt_sample

src_final = out_domain_data_src + in_domain_data_src_final
tgt_final = out_domain_data_tgt + in_domain_data_tgt_final

os.makedirs(out_dir, exist_ok=True)
open(os.path.join(out_dir,
                  'train.{}'.format(src_lang)), 'w').write(''.join(src_final))
open(os.path.join(out_dir,
                  'train.{}'.format(tgt_lang)), 'w').write(''.join(tgt_final))