# -*- coding: utf-8 -*-
from functools import reduce
from operator import add
import numpy as np
import pickle

try:
  append_eos
except NameError:
  from colab_function import append_eos, text_to_vec, train_data_to_train_and_target_vector, build_model, train
  from colab_function import predict, max_cos_sim, find_char_by_vec

#------------------------------------------------------------
# 06. Open training data.
#------------------------------------------------------------
# End of string of training texts.
EOS: str = '\t'
TRAIN_CHUNK_NUM = 8

# TODO: 全件 (names_has_uu) を回すとなると複数回に学習 (fit) を分ける必要がある
# output_training_data_file_name = 'names_gt_1000.pkl'
# with open('drive/names_gt_1000.tsv', 'r') as f:
output_training_data_file_name = 'names_has_uu.pkl'
with open('drive/names_has_uu.tsv', 'r') as f:
  raw_data = f.readlines()
raw_data = [line.strip() for line in raw_data]

raw_data_chunks = [raw_data[idx:idx + TRAIN_CHUNK_NUM] for idx in range(0, len(raw_data), TRAIN_CHUNK_NUM)]

with open('drive/term_vec.pkl', 'rb') as f:
  term_vec = pickle.load(f)

# add EOS
term_vec[EOS] = term_vec['</s>']

#------------------------------------------------------------
# 09. Preprocess to train.
# 10. Run training.
# 11. Save trained model.
#------------------------------------------------------------
try:
  text_vectors_list
except NameError:
  with open('drive/' + output_training_data_file_name, 'rb') as f:
    text_vectors_list = pickle.load(f)


vocabulary_size = len(term_vec)
combined_texts_list = [reduce(add, append_eos(chunk, eos=EOS)) for chunk in raw_data_chunks]
max_length = max(len(combined_texts) for combined_texts in combined_texts_list)

def fill_with_eos(text, length, eos):
  if len(text) >= length:
    return text
  filling_length = length - len(text)
  return text + eos * filling_length


combined_texts_list = [fill_with_eos(combined_texts, length=max_length, eos=EOS) for combined_texts in combined_texts_list]

text_vectors_list = [text_to_vec(combined_text, term_vec) for combined_text in combined_texts_list]
with open('drive/' + output_training_data_file_name, 'wb') as f:
  f.write(pickle.dumps(text_vectors_list))

# train_vectors, target_vectors = train_data_to_train_and_target_vector(text_vectors)
train_target_vectors_pair_list = [
  train_data_to_train_and_target_vector(text_vectors) for text_vectors in text_vectors_list
]

model = build_model(*train_target_vectors_pair_list[0])

for train_vectors, target_vectors in train_target_vectors_pair_list:
  model = train(np.array(train_vectors), np.array(target_vectors), model, epochs=50)
  with open('drive/names_has_uu_model.pkl', 'wb') as f:
  # with open('drive/names_gt_1000_model.pkl', 'wb') as f:
    f.write(pickle.dumps(model))



#------------------------------------------------------------
# 13. Prediction
#------------------------------------------------------------
# predict
input_str = '京'
input_vector = np.array([term_vec[char] for char in list(input_str)]).reshape((len(input_str), 1, 50))

# reshape to input dim matrix
output_vector = predict(input_vector, model)
print(output_vector.shape)

predicted_vector = max_cos_sim(output_vector[-1], text_vectors_list)

predicted_char = find_char_by_vec(predicted_vector, term_vec)
print(predicted_char)