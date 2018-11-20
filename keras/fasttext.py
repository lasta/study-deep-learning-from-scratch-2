# -*- coding: utf-8 -*-
import os
import sys
import glob
import math
import json
from pathlib import Path
import pickle

INPUT_FILE = 'data/names_has_uu.tsv'
TOKENIZED_FILE = '/tmp/tokenized_data'


def tokenize(file_name=INPUT_FILE):
    tokenized_lines = []
    with open(file_name, 'r') as input_file:
        tokenized_lines = [' '.join(list(line)) for line in input_file]
    with open(TOKENIZED_FILE, 'w') as tokenized_file:
        tokenized_file.writelines(tokenized_lines)


def dump():
    tokenize()
    os.system(f"fasttext cbow -input {TOKENIZED_FILE} -output model -minCount 1 -dim 50 -epoch 200 -lr 0.01")
    term_vec = {}

    with open('model.vec', 'r') as model_file:
        next(model_file)
        for line in model_file:
            entries = line.strip().split(' ')
            term = entries[0]
            vector = list(map(float, entries[1:]))
            term_vec[term] = vector
        with open('term_vec.pkl', 'wb') as dump_file:
            dump_file.write(pickle.dumps(term_vec))


def test():
    with open('term_vec.pkl', 'rb') as dumped_file:
        term_vec = pickle.load(dumped_file)
    
    print(term_vec['東'])

if __name__ == '__main__':
    dump()
    test()
