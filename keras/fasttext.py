# -*- coding: utf-8 -*-
import os
import sys
import glob
import math
import json
from pathlib import Path
import pickle

def dump():
    os.system("fasttext cbow -input data/names_has_uu.tsv -output model -minCount 2 -dim 300")
    term_vec = {}

    with open('model.vec', 'r') as model_file:
        next(model_file)
        for line in model_file:
            entries = line.split()
            term = ' '.join(entries[:-256])
            vector = list(map(float, entries[-256:]))
            term_vec[term] = vector
        with open('term_vec.pkl', 'wb') as dump_file:
            dump_file.write(pickle.dumps(term_vec))

if __name__ == '__main__':
    dump()