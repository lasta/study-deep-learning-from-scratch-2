# -*- coding: utf-8 -*-
import codecs
from preprocess import preprocess

# 431 rows
raw_train_data_path = './data/names_gt_1000.tsv'
# 178362 rows
# raw_train_data_path = './data/names_has_uu.tsv'

with codecs.open(raw_train_data_path, 'r', 'utf-8') as file:
    train_texts = [text.strip() for text in file.readlines()]

corpus, char_to_id, id_to_char = preprocess(train_texts)

vocabulary_size = len(char_to_id)

input = []
target = []

for i in range(0, vocabulary_size):
    input.append(train_texts)

# TODO: ここから先どうする?
#       seq2seq できるかな? (逆順も)