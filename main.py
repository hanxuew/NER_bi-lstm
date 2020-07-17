# -*- coding:utf-8 -*-
import logging
import os

import numpy as np

from data import read_corpus, read_dictionary
from model import BiLSTM_CRF, Config
from utils import NerCfgData

ner_cfg = NerCfgData()
label2id = ner_cfg.generate_tag_to_label()

logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))

## get char embeddings
word2id_pos2id = read_dictionary('word2id_pos2id_new.pkl')
word2id = word2id_pos2id['word2id']
pos2id = word2id_pos2id['pos2id']
word_embedding = np.array(np.load('word2vec.npy'), dtype=np.float32)
pos_embedding = np.array(np.load('pos2vec.npy'), dtype=np.float32)

config = Config(word2id, pos2id, label2id, batch_size=128, n_epochs=200, n_neurons=60)
config.word_embedding = word_embedding
config.pos_embedding = pos_embedding

## read corpus and get training data
train_data, test_data = read_corpus('train_data')
# test_data = read_corpus('test_data')
# test_size = len(test_data)

model = BiLSTM_CRF(is_training=True, config=config)
model.build_graph()
model.train(train_data=train_data, valid_data=test_data)
# model.test(test_data)
