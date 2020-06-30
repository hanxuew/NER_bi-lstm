# -*- coding:utf-8 -*-
import logging
import os
import pickle as pk
import numpy as np

from data import read_corpus, read_dictionary
from model import BiLSTM_CRF, Config
from utils import NerCfgData

from random import random, shuffle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ner_cfg = NerCfgData()
label2id = ner_cfg.generate_tag_to_label()

logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))

## get char embeddings
word2id_pos2id = read_dictionary('word2id_pos2id_new.pkl')
# word2id = word2id_pos2id['word2id']
word2id = read_dictionary('word2id.pk')
word2id['<UNK>'] = len(word2id.items())
pos2id = word2id_pos2id['pos2id']

# word_embedding = np.array(np.load('word2vec.npy'), dtype=np.float32)
# word_embedding = np.array(np.load('word_embedding.pk'), dtype=np.float32)
with open('./word_embedding.pk', 'rb') as file:
    word_emb = pk.load(file)
word_embedding = np.concatenate((np.array(word_emb, dtype=np.float32), np.array([[random() for i in range(200)]])))
pos_embedding_temp = np.array(np.load('pos2vec.npy'), dtype=np.float32)
pos_embedding = np.concatenate((pos_embedding_temp, np.array([[random() for i in range(10)]])))
config = Config(word2id, pos2id, label2id, batch_size=128, n_epochs=200, n_neurons=60)
config.word_embedding = word_embedding
config.pos_embedding = pos_embedding

## read corpus and get training data
# total_data = read_corpus('train_data_')
# shuffle(total_data)
# test_data = total_data[:20000]
# train_data = total_data[20000:]
test_data = read_corpus('test_data')
# test_data = read_corpus('test_data')
# test_size = len(test_data)
model = BiLSTM_CRF(is_training=True, config=config)
model.build_graph()
# model.train(train_data=train_data, valid_data=test_data)
model.test(test_data)
