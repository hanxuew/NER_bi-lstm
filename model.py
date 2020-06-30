# -*- coding:utf-8 -*-
import time
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from utils import next_batch, pad_seqs, get_tmp_file_name, write_result, compute_prf_score

class Config(object):
    def __init__(self, word2id, pos2id, label2id, batch_size, n_epochs, n_neurons, is_training=True):
        self.word_vocab = word2id
        self.pos_vocab = pos2id
        self.label_vocab = label2id
        self.summary_path = 'log'
        self.model_checkpoint_path = 'checkpoint/ner_model.ckpt'
        self.model_directory = 'checkpoint/'
        self.is_training = is_training
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_neurons = n_neurons
        self.n_classes = len(label2id)
        self.n_dims = 40
        self.n_word_dims = 30
        self.n_pos_dims = 10

        self.learning_rate = 0.001
        self.keep_prob = 0.8
        self.max_grad_norm = 5.0

        self.word_embedding = None
        self.pos_embedding = None


class BiLSTM_CRF(object):
    def __init__(self, is_training, config):
        self.word_vocab = config.word_vocab
        self.pos_vocab = config.pos_vocab
        self.label_vocab = config.label_vocab
        self.id2word = dict(zip(self.word_vocab.values(), self.word_vocab.keys()))
        self.id2pos = dict(zip(self.pos_vocab.values(), self.pos_vocab.keys()))
        self.id2label = dict(zip(self.label_vocab.values(), self.label_vocab.keys()))
        self.summary_path = config.summary_path
        self.model_checkpoint_path = config.model_checkpoint_path
        self.model_directory = config.model_directory

        self.is_training = is_training
        self.batch_size = config.batch_size
        self.n_epochs = config.n_epochs
        self.n_neurons = config.n_neurons
        self.n_classes = config.n_classes
        self.n_dims = config.n_dims
        self.n_word_dims = config.n_word_dims
        self.n_pos_dims = config.n_pos_dims

        self.learning_rate = config.learning_rate
        self.keep_prob = config.keep_prob
        self.max_grad_norm = config.max_grad_norm

        self.word_embedding = config.word_embedding
        self.pos_embedding = config.pos_embedding

        self.best_f1 = 0.9
        self.best_epoch = -1
        self.graph = tf.Graph()
        # self.project = projector.ProjectorConfig()

        self.save_flag = True #False

    def build_graph(self):
        with self.graph.as_default():
            self._add_placeholder()
            self._gpu_set_config()
            self._add_embedding()
            self._add_bi_rnn()
            self._add_loss_op()
            self._add_train_op()
            self._add_predict_op()
            self._add_saver()
            self._add_init_op()

    def _gpu_set_config(self):
        self.gpu_config = tf.ConfigProto()
        self.gpu_config.gpu_options.allow_growth = True
        self.gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.9

    def _add_placeholder(self):
        with tf.name_scope('placeholder'):
            self.word_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_inputs')
            self.pos_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='pos_inputs')
            self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='targets')
            self.batch_sequences_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_length')
            self.keep_prob_pl = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def _add_embedding(self):
        with tf.name_scope('embedding'):

            if self.word_embedding is None:
                self.word_embedding = tf.get_variable(name='word_embedding',
                                                      shape=[len(self.word_vocab), self.n_word_dims], dtype=tf.float32,
                                                      initializer=tf.contrib.layers.variance_scaling_initializer())
            else:
                self.word_embedding = tf.Variable(self.word_embedding, dtype=tf.float32, name='word_embedding')
            if self.pos_embedding is None:
                self.pos_embedding = tf.get_variable(name='pos_embedding', shape=[len(self.pos_vocab), self.n_pos_dims],
                                                     dtype=tf.float32,
                                                     initializer=tf.contrib.layers.variance_scaling_initializer())
            else:
                self.pos_embedding = tf.Variable(self.pos_embedding, dtype=tf.float32, name='pos_embedding')

            word_embedding_inputs = tf.nn.embedding_lookup(self.word_embedding, self.word_inputs)
            pos_embedding_inputs = tf.nn.embedding_lookup(self.pos_embedding, self.pos_inputs)

            # self.projector_embedding = self.project.embeddings.add()
            # self.projector_embedding.tensor_name = self.word_embedding.name
            # self.projector_embedding.metadata_path = 'metadata.tsv'

            if self.is_training:
                # word_embedding_inputs_ = tf.nn.dropout(word_embedding_inputs, keep_prob=self.keep_prob)
                # pos_embedding_inputs_ = tf.nn.dropout(pos_embedding_inputs, keep_prob=self.keep_prob)
                inputs = tf.concat([word_embedding_inputs, pos_embedding_inputs], axis=-1)
                # self.inputs = inputs
                self.inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob, name='dropout_inputs')
            else:
                self.inputs = tf.concat([word_embedding_inputs, pos_embedding_inputs], axis=-1, name='inputs')

    def _add_bi_rnn(self):
        with tf.name_scope('bi_lstm'):
            forward_cell = tf.contrib.rnn.LSTMCell(self.n_neurons, activation=tf.nn.elu)
            backward_cell = tf.contrib.rnn.LSTMCell(self.n_neurons, activation=tf.nn.elu)
            (forward_outputs, backward_outputs), (_, _) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward_cell,
                cell_bw=backward_cell,
                inputs=self.inputs,
                sequence_length=self.batch_sequences_length,
                dtype=tf.float32
            )
            lstm_outputs_ = tf.concat([forward_outputs, backward_outputs], axis=-1, name='lstm_output')
            # self.lstm_outputs = tf.nn.dropout(lstm_outputs_, keep_prob=self.keep_prob)
            # self.lstm_input_output = tf.concat([self.lstm_outputs, self.inputs], axis=-1, name='lstm_input_output')
            self.outputs = lstm_outputs_

        # with tf.name_scope('add_lstm'):
        #     cell = tf.contrib.rnn.LSTMCell(2 * self.n_neurons, activation=tf.nn.elu)
        #     outputs_, _ = tf.nn.dynamic_rnn(
        #         cell=cell,
        #         inputs=self.lstm_outputs,
        #         sequence_length=self.batch_sequences_length,
        #         dtype=tf.float32
        #     )
        #     self.outputs = tf.nn.dropout(outputs_, keep_prob=self.keep_prob)

        with tf.name_scope('bi_lstm_proj'):
            proj_w = tf.get_variable(name='proj_w', shape=[2 * self.n_neurons, self.n_classes], dtype=tf.float32,
                                     initializer=tf.contrib.layers.variance_scaling_initializer(),
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))
            # proj_w = tf.get_variable(name='proj_w', shape=[2 * self.n_neurons + self.n_dims, self.n_classes],
            #                          dtype=tf.float32,
            #                          initializer=tf.contrib.layers.variance_scaling_initializer(),
            #                          regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
            proj_b = tf.get_variable(name='proj_b', shape=[self.n_classes], dtype=tf.float32,
                                     initializer=tf.zeros_initializer())

            shape_list = tf.shape(self.outputs)
            logits = tf.nn.xw_plus_b(tf.reshape(self.outputs, shape=[-1, 2 * self.n_neurons]),
                                     proj_w, proj_b)
            self.logits = tf.reshape(logits, shape=[-1, shape_list[1], self.n_classes], name='logits')

    def _add_loss_op(self):
        with tf.name_scope('loss'):
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits, tag_indices=self.targets,
                                                                        sequence_lengths=self.batch_sequences_length)
            self.likelihood_loss = tf.reduce_mean(-log_likelihood)
            self.l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')
            # self.l2_loss = tf.contrib.layers.apply_regularization(
            #     regularizer=tf.contrib.layers.l2_regularizer(scale=0.001),
            #     weights_list=tf.trainable_variables())
            # need consider log
            self.loss = self.likelihood_loss + self.l2_loss

            tf.summary.scalar(name='likelihood_loss', tensor=self.likelihood_loss)
            tf.summary.scalar(name='l2 loss', tensor=self.l2_loss)
            tf.summary.scalar(name='loss', tensor=self.loss)

    def _add_predict_op(self):
        with tf.name_scope('predict'):
            self.predict = tf.nn.softmax(self.logits, dim=-1)

    def _add_train_op(self):
        with tf.name_scope('train_op'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_value(grad, -self.max_grad_norm, self.max_grad_norm), var) for grad, var in
                             grads_and_vars]
            self.train_op = optimizer.apply_gradients(clipped_grads,
                                                      global_step=self.global_step)

    def _add_summary_op(self, sess):
        self.summary_merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)
        # projector.visualize_embeddings(self.file_writer, self.project)

    def _add_init_op(self):
        self.init_op = tf.global_variables_initializer()

    def _add_saver(self):
        self.saver = tf.train.Saver()

    def _get_feed_dict(self, batch_words, batch_poses, batch_labels=None, training_flag=True):
        feed_dict = {}
        batch_pad_words, batch_words_len = pad_seqs(batch_words)
        batch_pad_poses, batch_poses_len = pad_seqs(batch_poses)
        feed_dict[self.word_inputs] = batch_pad_words
        feed_dict[self.pos_inputs] = batch_pad_poses
        feed_dict[self.batch_sequences_length] = batch_words_len
        if batch_labels:
            batch_pad_labels, _ = pad_seqs(batch_labels)
            feed_dict[self.targets] = batch_pad_labels
        if training_flag:
            feed_dict[self.keep_prob_pl] = self.keep_prob
        else:
            feed_dict[self.keep_prob_pl] = 1.0
        return feed_dict, batch_words_len

    def _run_one_epoch(self, sess, train_data, valid_data, epoch, saver):
        save_flag = True
		# False
        start_time = time.time()
        data_len = len(train_data)
        num_batches = (data_len + self.batch_size - 1) // self.batch_size
        batches = next_batch(train_data,
                             (self.word_vocab, self.pos_vocab, self.label_vocab),
                             self.batch_size, shuffle=False)
        valid_start_index = epoch * self.batch_size
        valid_end_index = (epoch + 1) * self.batch_size
        if valid_start_index >= len(train_data):
            valid_start_index = 0
            valid_end_index = self.batch_size
        else:
            if valid_end_index <= len(train_data):
                valid_end_index = (epoch + 1) * self.batch_size
            else:
                valid_end_index = -1
        train_valid_data = train_data[valid_start_index:valid_end_index]
        for step, (batch_words, batch_poses, batch_labels) in enumerate(batches):
            real_step = epoch * num_batches + step + 1
            feed_dict, _ = self._get_feed_dict(batch_words, batch_poses, batch_labels)
            _, loss, summary, global_step = sess.run([self.train_op, self.loss, self.summary_merged, self.global_step],
                                                     feed_dict=feed_dict)
            self.file_writer.add_summary(summary, global_step=global_step)
            if epoch >= 0 and self.save_flag:
                saver.save(sess, self.model_checkpoint_path, global_step=real_step)
                self.save_flag = False
            if step + 1 == num_batches and epoch > 0:
                train_predict_label_list, _ = self.valid(sess, train_valid_data)
                tmp_file_path = get_tmp_file_name('output')
                write_result(train_valid_data, train_predict_label_list, self.id2label, tmp_file_path)
                avg_f1 = compute_prf_score(tmp_file_path)
                if avg_f1 > self.best_f1 and avg_f1 > 0.9:
                    self.save_flag = True
                print('TRAIN: epoch {}, avg f1 {}\n'.format(epoch, avg_f1))

        predict_label_list, _ = self.valid(sess, valid_data, training_flag=False)
        tmp_file_path = get_tmp_file_name('output')
        write_result(valid_data, predict_label_list, self.id2label, tmp_file_path)
        avg_f1 = compute_prf_score(tmp_file_path)
        if self.best_f1 < avg_f1:
            self.best_f1 = avg_f1
            self.best_epoch = epoch
            self.save_flag = True
        print('TEST: epoch {}, cost time is {}, avg f1 {}\n'.format(epoch, time.time() - start_time, avg_f1))

    def train(self, train_data, valid_data):
        with tf.Session(graph=self.graph, config=self.gpu_config) as sess:
            self._add_summary_op(sess)
            sess.run([self.init_op])
            for epoch in range(self.n_epochs):
                self._run_one_epoch(sess, train_data, valid_data, epoch, self.saver)
            print('best avg f1={}, at epoch={}'.format(self.best_f1, self.best_epoch))
        pass

    def test(self, data):
        with tf.Session(graph=self.graph) as sess:
            ckpt_path = tf.train.latest_checkpoint(self.model_directory)
            self.saver.restore(sess, ckpt_path)
            self.is_training = False
            predict_label_list, _ = self.valid(sess, data)
            tmp_file_path = get_tmp_file_name('output')
            write_result(data, predict_label_list, self.id2label, tmp_file_path)
            avg_f1 = compute_prf_score(tmp_file_path)
            print('avg_f1', avg_f1)
            print('predict done!!!')

        pass

    def valid(self, sess, data, training_flag=True):
        label_list, seq_len_list = [], []
        for batch_words, batch_poses, batch_labels in next_batch(data,
                                                                 (self.word_vocab, self.pos_vocab, self.label_vocab),
                                                                 self.batch_size, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, batch_words, batch_poses,
                                                                training_flag=training_flag)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, batch_words, batch_poses, training_flag=True):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self._get_feed_dict(batch_words, batch_poses, training_flag=training_flag)

        logits, transition_params = sess.run([self.logits, self.transition_params],
                                             feed_dict=feed_dict)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_list):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        return label_list, seq_len_list
