import sys
import os

file_path = sys.argv[1]

num_words = 0
num_sents = 0
with open(file_path) as ifile:
    for line in ifile:
        num_sents += 1
        num_words += len(line.strip().split())

print(file_path, num_words, num_sents, 1. * num_words / num_sents)