# -*- coding: utf-8 -*-
import codecs
from preprocess import preprocess, text_to_vec, convert_one_hot

# 431 rows
raw_train_data_path = './data/names_gt_1000.tsv'
# 178362 rows
# raw_train_data_path = './data/names_has_uu.tsv'

with codecs.open(raw_train_data_path, 'r', 'utf-8') as file:
    train_texts = [text.strip() for text in file.readlines()]

corpus, char_to_id, id_to_char = preprocess(train_texts)
vocabulary_size = len(char_to_id)

text_vectors = [text_to_vec(text, char_to_id) for text in train_texts]
text_one_hot_vectors = [convert_one_hot(vec, vocabulary_size) for vec in text_vectors]

print(text_vectors[2])
print(text_one_hot_vectors[2][0])
