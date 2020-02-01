import json
import os
import random

data_dir = '/data/medg/misc/jindi/nlp/datasets/OPUS/TED_mono'
en_data = json.load(open(os.path.join(data_dir, 'ted_clean_transcripts_en.json')))
de_data = json.load(open(os.path.join(data_dir, 'ted_clean_transcripts_de.json')))
ro_data = json.load(open(os.path.join(data_dir, 'ted_clean_transcripts_ro.json')))

def text_preprocessing(text):
    return [line.lower() for line in text if len(line.split()) <= 150]


def prepare_mono_corpora(data_1, data_2):
    lang_1 = list(data_1.keys())[0][-2:]
    lang_2 = list(data_2.keys())[0][-2:]
    links_1 = set([line[:-2] for line in data_1.keys()])
    links_2 = set([line[:-2] for line in data_2.keys()])
    final_links_1 = list(links_1.difference(links_2))
    final_links_2 = list(links_2.difference(links_1))
    common_links = list(links_1.intersection(links_2))
    num_1 = (len(common_links) + len(final_links_2) - len(final_links_1)) // 2
    random.seed(1234)
    idx_ls = set(random.sample(range(len(common_links)), num_1))
    final_links_1 += [common_links[idx] for idx in range(len(common_links)) if idx in idx_ls]
    final_links_2 += [common_links[idx] for idx in range(len(common_links)) if idx not in idx_ls]
    assert abs(len(final_links_1) - len(final_links_2)) <= 1
    corpus_1 = text_preprocessing([line for link in final_links_1 for line in data_1[link+lang_1]])
    corpus_2 = text_preprocessing([line for link in final_links_2 for line in data_2[link+lang_2]])
    return corpus_1, corpus_2

out_dir = '/data/medg/misc/jindi/nlp/datasets/OPUS'

os.makedirs(os.path.join(out_dir, 'de-en', 'TED_mono'), exist_ok=True)
en_corpus, de_corpus = prepare_mono_corpora(en_data, de_data)
open(os.path.join(out_dir, 'de-en', 'TED_mono', 'train.de.mono'), 'w').write('\n'.join(de_corpus))
open(os.path.join(out_dir, 'de-en', 'TED_mono', 'train.en.mono'), 'w').write('\n'.join(en_corpus))

os.makedirs(os.path.join(out_dir, 'en-ro', 'TED_mono'), exist_ok=True)
en_corpus, ro_corpus = prepare_mono_corpora(en_data, ro_data)
open(os.path.join(out_dir, 'en-ro', 'TED_mono', 'train.ro.mono'), 'w').write('\n'.join(ro_corpus))
open(os.path.join(out_dir, 'en-ro', 'TED_mono', 'train.en.mono'), 'w').write('\n'.join(en_corpus))