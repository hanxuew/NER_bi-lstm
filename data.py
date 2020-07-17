# -*- coding:utf-8 -*-
import os
import pickle
import numpy as np

# label2id = {'o': 0,
#             'b_srs': 1,
#             'b_as': 2,
#             'b_acr': 3,
#             'b_exm': 4,
#             'b_exa': 5,
#             'b_rv': 6,
#             'b_dis': 7,
#             'b_tm': 8,
#             'b_dat': 9,
#             'b_prw': 10,
#             'b_saw': 11,
#             'b_sc': 12,
#             'i_srs': 13,
#             'i_as': 14,
#             'i_acr': 15,
#             'i_exm': 16,
#             'i_exa': 17,
#             'i_rv': 18,
#             'i_dis': 19,
#             'i_tm': 20,
#             'i_dat': 21,
#             'i_prw': 22,
#             'i_saw': 23,
#             'i_sc': 24,
#             'e_srs': 25,
#             'e_as': 26,
#             'e_acr': 27,
#             'e_exm': 28,
#             'e_exa': 29,
#             'e_rv': 30,
#             'e_dis': 31,
#             'e_tm': 32,
#             'e_dat': 33,
#             'e_prw': 34,
#             'e_saw': 35,
#             'e_sc': 36,
#             's_srs': 37,
#             's_as': 38,
#             's_acr': 39,
#             's_exm': 40,
#             's_exa': 41,
#             's_rv': 42,
#             's_dis': 43,
#             's_tm': 44,
#             's_dat': 45,
#             's_prw': 46,
#             's_saw': 47,
#             's_sc': 48
#             }
label2id = {}

def read_dictionary(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word_pos_vocab = pickle.load(fr)
    return word_pos_vocab


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, pos_, tag_ = [], [], []
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            word_lable_arr = line.split()
            word = word_lable_arr[0]
            pos = word_lable_arr[1]
            label = word_lable_arr[-1]
            sent_.append(word)
            pos_.append(pos)
            tag_.append(label)
        else:
            data.append((sent_, pos_, tag_))
            sent_, pos_, tag_ = [], [], []

    if len(sent_) > 0:
        data.append((sent_, pos_, tag_))

    return data


def vocab_build(corpus_path, vocab_path, min_count):
    """
    :param corpus_path:
    :param vocab_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    pos2id = {}

    for sent_, pos_, tag_ in data:
        for word in sent_:
            # if word.isdigit():
            #    word = '<NUM>'
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1
        for pos in pos_:
            if pos not in pos2id:
                pos2id[pos] = len(pos2id)

    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 0
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    pos2id['<UNK>'] = len(pos2id)
    print(word2id['<UNK>'])
    # word2id['<PAD>'] = 0

    word_pos_vocab = {'word2id': word2id, 'pos2id': pos2id}
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word_pos_vocab, fw)

    return word_pos_vocab


def random_embedding(vocab, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat
if __name__ == '__main__':
    vocab_build('train_data', 'word2id_pos2id_new.pkl', min_count=5)