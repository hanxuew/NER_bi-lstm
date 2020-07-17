from data import read_corpus, read_dictionary
import numpy as np
word2id_pos2id = read_dictionary('word2id_pos2id_new.pkl')

# print(word2id_pos2id['word2id'])
# print(word2id_pos2id['pos2id'])
data = read_corpus('train_data')

for i in data:
    for j in i[1]:
        if j not in word2id_pos2id['pos2id'].keys():
            break
            print(j, 'is not in dict')
print('done...')

# pos_embedding = np.array(np.load('pos2vec.npy'), dtype=np.float32)
# train_data = read_corpus('train_data')
# test_data = read_corpus('test_data')
# for i in train_data:
#     print(i)
#     if len(i) == 1:
#         print('aha')