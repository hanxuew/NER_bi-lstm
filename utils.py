# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os, sys, configparser, codecs, logging, traceback, datetime, random
import re
from collections import Counter, defaultdict
import csv
from copy import deepcopy

current_dir_name = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)


def seq2id(sent, vocab):
    sentence_id = []
    for word in sent:
        # if word.isdigit():
        #    word = '<NUM>'
        if word not in vocab:
            word = '<UNK>'
        sentence_id.append(vocab[word])
    return sentence_id


def pos_sentence2id(pos_sent, pos2id):
    pos_sentence_id = []
    for pos in pos_sent:
        if pos not in pos2id:
            pos = '<UNK>'
        pos_sentence_id.append(pos2id[pos])
    return pos_sentence_id


def next_batch(data, vocab, batch_size=64, shuffle=False):
    # data_indices = range(len(data))
    if shuffle:
        np.random.shuffle(data)
    batch_words, batch_poses, batch_targets = [], [], []
    word2id, pos2id, label2id = vocab
    for sent_seq, pos_seq, label_seq in data:
        tmp_sent_id_seq = seq2id(sent_seq, word2id)
        tmp_pos_id_seq = seq2id(pos_seq, pos2id)
        tmp_label_id_seq = [label2id[label] for label in label_seq]

        if batch_size == len(batch_words):
            yield batch_words, batch_poses, batch_targets
            batch_words, batch_poses, batch_targets = [], [], []

        batch_words.append(tmp_sent_id_seq)
        batch_poses.append(tmp_pos_id_seq)
        batch_targets.append(tmp_label_id_seq)

    if len(batch_words) > 0:
        yield batch_words, batch_poses, batch_targets
        batch_words, batch_poses, batch_targets = [], [], []


def pad_seqs(seqs, padding_marker=0):
    max_len = max(map(lambda item: len(item), seqs))
    pad_seqs_result, seqs_len_list = [], []
    for seq in seqs:
        seq_len = len(seq)
        pad_seq = seq[:max_len] + [padding_marker] * max([max_len - seq_len, 0])
        pad_seqs_result.append(pad_seq)
        seqs_len_list.append(seq_len)
    return pad_seqs_result, seqs_len_list


def comput_f_val(file_path):
    """
    :param file_path:
    :return:
    """
    ner_cfg = NerCfgData()
    input_file = open(file_path, "r", buffering=2 << 10, encoding="utf8")

    lines = input_file.readlines()
    input_file.close()
    i = 0

    matrix = {}
    entity_type_map = {}
    entity_type_cnt = {}  # 统计各类实体的个数，按真实类型计算
    entity_extra_oth_cnt = 0
    f_dict = {}
    while i < len(lines):
        line = lines[i].strip()

        if '' == line:
            i += 1
            continue
        else:
            if i + 1 >= len(lines):
                break
            prict_tag = lines[i + 1].strip()
            i += 1

            one_predict = line
            array = one_predict.split("\t")

            if "o" == array[2]:
                entity_type_map["o"] = "o"
                try:
                    if "_" in array[3]:
                        # true to preditct
                        t_2_p = array[2] + "___" + array[3].split("_")[1]
                        if t_2_p in matrix:
                            matrix[t_2_p] = matrix[t_2_p] + 1
                        else:
                            matrix[t_2_p] = 1
                    else:
                        t_2_p = array[2] + "___" + array[3]
                        if t_2_p in matrix:
                            matrix[t_2_p] = matrix[t_2_p] + 1
                        else:
                            matrix[t_2_p] = 1
                except Exception as e:
                    logger.info(e)

            # s开头 单独成实体的情况
            elif array[2].startswith("s_"):
                entity_type_map[array[2].split("_")[1]] = array[2].split("_")[1]
                entity_type = array[2].split("_")[1]
                entity_type_cnt[entity_type] = 1 if entity_type not in entity_type_cnt.keys() else entity_type_cnt[
                                                                                                       entity_type] + 1
                entity_extra_oth_cnt += 1

                if "_" in array[3]:
                    entity_type_map[array[3].split("_")[1]] = array[3].split("_")[1]
                    t_2_p = array[2].split("_")[1] + "___" + array[3].split("_")[1]
                    if t_2_p in matrix:
                        matrix[t_2_p] = matrix[t_2_p] + 1
                    else:
                        matrix[t_2_p] = 1
                else:
                    t_2_p = array[2].split("_")[1] + "___" + array[3]
                    if t_2_p in matrix:
                        matrix[t_2_p] = matrix[t_2_p] + 1
                    else:
                        matrix[t_2_p] = 1
            else:
                # 以b开头的实体，意味接下来是一个实体
                entity_type_map[array[2].split("_")[1]] = array[2].split("_")[1]
                entity_type = array[2].split("_")[1]
                entity_type_cnt[entity_type] = 1 if entity_type not in entity_type_cnt.keys() else entity_type_cnt[
                                                                                                       entity_type] + 1
                entity_extra_oth_cnt += 1
                try:
                    if "o" == array[3]:
                        entity_type_map[array[3]] = array[3]
                    else:
                        entity_type_map[array[3].split("_")[1]] = array[3].split("_")[1]
                    if array[2].startswith("b_"):
                        std_tag = array[2].split("_")[1]
                        pred_tag_map = {}
                        if "_" in array[3]:
                            pred_tag = array[3].split("_")[1]
                            if pred_tag in pred_tag_map:
                                pred_tag_map[pred_tag] = pred_tag_map[pred_tag] + 1
                            else:
                                pred_tag_map[pred_tag] = 1

                        else:
                            pass
                        # 在b后找全实体
                        while not array[2].startswith("e_") and array[2].split("_")[1] == std_tag and i < len(lines):
                            line = lines[i].replace("\n", "").strip()
                            if '' == line.strip() or line.strip() is None:
                                i = i + 1
                                continue
                            i = i + 1

                            one_predict = line
                            array = one_predict.split("\t")

                            try:
                                entity_type_map[array[2].split("_")[1]] = array[2].split("_")[1]
                                if "o" == array[3]:
                                    entity_type_map[array[3]] = array[3]
                                else:
                                    entity_type_map[array[3].split("_")[1]] = array[3].split("_")[1]
                            except Exception as e:
                                logger.info("file_path={}, line={}, i={}, {}".format(file_path, line, i, e))
                                break

                            if "_" in array[3]:
                                pred_tag = array[3].split("_")[1]
                                if pred_tag in pred_tag_map:
                                    pred_tag_map[pred_tag] = pred_tag_map[pred_tag] + 1
                                else:
                                    pred_tag_map[pred_tag] = 1
                        if array[2].startswith("e_"):
                            if 0 == len(pred_tag_map):
                                t_2_p = std_tag + "___o"
                                if t_2_p in matrix:
                                    matrix[t_2_p] = matrix[t_2_p] + 1
                                else:
                                    matrix[t_2_p] = 1
                            elif 1 == len(pred_tag_map):
                                # 预测出的标签只有一种类型
                                for k, v in pred_tag_map.items():
                                    t_2_p = std_tag + "___" + k
                                    if t_2_p in matrix:
                                        matrix[t_2_p] = matrix[t_2_p] + 1
                                    else:
                                        matrix[t_2_p] = 1
                            # 预测出的标签有多种类型
                            else:
                                # 以出现次数最多，但不是"o"的标签，做为预测的最终标签
                                the_current_tag = ""
                                the_current_tag_size = -100
                                for k, v in pred_tag_map.items():
                                    if v > the_current_tag_size:
                                        the_current_tag = k
                                t_2_p = std_tag + "___" + the_current_tag
                                if t_2_p in matrix:
                                    matrix[t_2_p] = matrix[t_2_p] + 1
                                else:
                                    matrix[t_2_p] = 1
                        else:
                            logger.info("error:行：i=" + str(i) + ",训练数据的标注有错误。one_predict=" + one_predict)
                            logger.info(line + prict_tag)
                except Exception as e:
                    logger.info("####{}####".format(array))
                    logger.info("{}, {}".format(e, traceback.format_exc()))

    # 打印混淆矩阵

    # 打印表头
    table_head = "\t"
    for k, v in entity_type_map.items():
        if "o" == k:
            table_head = table_head + "其他" + "\t"
        else:
            logger.info(k)
            if k in ner_cfg.ner_tags:
                table_head = table_head + ner_cfg.ner_tags[k] + "\t"
            else:
                table_head = table_head + k + "\t"
    print(table_head)

    # 打印值
    for k, v in entity_type_map.items():
        the_current_value = ""
        if "o" == k:
            the_current_value = the_current_value + "其他" + "\t"
        else:
            # the_current_value = ner_cfg.ner_tags[k] + "\t"
            if k in ner_cfg.ner_tags:
                the_current_value = ner_cfg.ner_tags[k] + "\t"
            else:
                the_current_value = the_current_value + k + "\t"
        for x, y in entity_type_map.items():
            k_x = k + "___" + x
            if k_x in matrix:
                the_current_value = the_current_value + str(matrix[k_x]) + "\t"
            else:
                the_current_value = the_current_value + "0\t"
        print(the_current_value)

    # 计算 p r f 值
    entity_type_real_count_map = {}
    entity_type_predicted_count_map = {}
    entity_type_predicted_and_right_count_map = {}

    for k, v in entity_type_map.items():
        for x, y in entity_type_map.items():
            k_x = k + "___" + x
            if k == x:
                if k_x in matrix:
                    entity_type_predicted_and_right_count_map[k] = matrix[k_x]

            if k in entity_type_real_count_map:
                if k_x in matrix:
                    entity_type_real_count_map[k] = entity_type_real_count_map[k] + matrix[k_x]
            else:
                if k_x in matrix:
                    entity_type_real_count_map[k] = matrix[k_x]
            x_k = x + "___" + k
            if k in entity_type_predicted_count_map:
                if x_k in matrix:
                    entity_type_predicted_count_map[k] = entity_type_predicted_count_map[k] + matrix[x_k]
            else:
                if x_k in matrix:
                    entity_type_predicted_count_map[k] = matrix[x_k]

    # 输出每个类的prf
    for k, v in entity_type_map.items():
        the_current_tag_cn_name = ""
        if "o" == k:
            the_current_tag_cn_name = "其他"
        else:
            if k in ner_cfg.ner_tags:
                the_current_tag_cn_name = ner_cfg.ner_tags[k]
            else:
                the_current_tag_cn_name = k

        if k not in entity_type_predicted_and_right_count_map:
            entity_type_predicted_and_right_count_map[k] = 0
        if k not in entity_type_predicted_count_map:
            entity_type_predicted_count_map[k] = 0.02
        if k not in entity_type_real_count_map:
            entity_type_real_count_map[k] = 0.02

        p = entity_type_predicted_and_right_count_map[k] / entity_type_predicted_count_map[k]
        r = entity_type_predicted_and_right_count_map[k] / entity_type_real_count_map[k]
        f = 0.0
        if 0 == (p + r):
            pass
        else:
            f = (2 * p * r / (p + r))
        if "o" != k:
            f_dict[k] = f

            print(the_current_tag_cn_name + "(%%):\tP=%.2f,\tR=%.2f,\tF=%.2f" % ((p * 100, r * 100, f * 100)))

    # 消除分母为0的情况
    entity_extra_oth_cnt = 1 if entity_extra_oth_cnt == 0 else entity_extra_oth_cnt
    weighted_f_val = 0
    for tmp_enty_type in f_dict.keys():
        tmp_percent = float(entity_type_cnt.get(tmp_enty_type, 0)) / entity_extra_oth_cnt
        weighted_f_val += tmp_percent * f_dict[tmp_enty_type]
    return weighted_f_val


def compute_prf_score(file_path):
    ner_cfg = NerCfgData()
    names_map = dict(ner_cfg.ner_tags)
    names_map['o'] = '其他'
    true_data, predict_data = read_predict_result_from_file(file_path)

    confusion_matrix, label2id = compute_confusion_matrix(true_data, predict_data, names_map)

    print_confusion_matrix(confusion_matrix, names_map, label2id)
    true_dict = count_labels_frequency(true_data)
    predict_dict = count_labels_frequency(predict_data)
    tp_dict = count_true_positive(true_data, predict_data)
    result = compute_prf_from_count_dict(tp_dict, true_dict, predict_dict)
    print_result(result, names_map)
    p = 0
    r = 0
    tp = np.sum([freq for label, freq in tp_dict.items() if label != 'o'])
    at = np.sum([freq for label, freq in true_dict.items() if label != 'o'])
    ap = np.sum([freq for label, freq in predict_dict.items() if label != 'o'])
    if at != 0:
        r = tp / at
    if ap != 0:
        p = tp / ap
    total_f = compute_f_value(p, r)
    return total_f


def print_result(data, labels_dict):
    for label, prf_value in data.items():
        print('{}(%)\t{}\t{}\t{}'.format(labels_dict.get(label, label), *prf_value))


def print_confusion_matrix(confusion_matrix, label_map, label2id):
    columns_name = label2id.keys()
    columns_len = len(columns_name)
    id2label = {indx: label for label, indx in label2id.items()}
    print('\t{}'.format('\t'.join([label_map.get(id2label.get(item)) for item in range(columns_len)])))
    for row_index, row_data in enumerate(confusion_matrix):
        print('{}\t{}'.format(label_map.get(id2label.get(row_index)), '\t'.join([str(item) for item in row_data])))


def read_predict_result_from_file(file):
    try:
        with open(file, 'r', encoding='utf8') as fp:
            source_data_df = pd.read_csv(fp, sep='\t', header=None, quoting=csv.QUOTE_NONE)
            true_data = [item for item in source_data_df[2].values]
            predict_data = [str(item).strip() for item in source_data_df[3].values]

            return true_data, predict_data
    except:
        with open(file, 'r', encoding='gbk') as fp:
            source_data_df = pd.read_csv(fp, sep='\t', header=None, quoting=csv.QUOTE_NONE)
            true_data = [item for item in source_data_df[2].values]
            predict_data = [str(item).strip() for item in source_data_df[3].values]

            return true_data, predict_data

def count_labels_frequency(data):
    process_data = process_tags(data)
    if process_data is None:
        return None
    counter = Counter(process_tags(data))
    return dict(counter)


def count_true_positive(true_data, predict_data):
    true_positive_dict = defaultdict(int)
    true_len = len(true_data)
    predict_len = len(predict_data)
    if true_len != predict_len:
        return true_positive_dict
    i = 0
    while i < true_len:
        if true_data[i] != predict_data[i]:
            i += 1
            continue
        else:
            if true_data[i].lower() == 'o':
                true_positive_dict[true_data[i]] += 1
                i += 1
            elif str(true_data[i]).lower().startswith('s_'):
                label = get_label_from_ann(true_data[i])
                true_positive_dict[label] += 1
                i += 1
            # 确保将整个实体遍历完，即遍历到e_*
            elif str(true_data[i]).lower().startswith('b_'):
                label = get_label_from_ann(true_data[i])
                entity_equal_flag = True
                j = i + 1
                while j < true_len:
                    if true_data[j] != predict_data[j]:
                        if entity_equal_flag:
                            entity_equal_flag = False

                    if str(true_data[j]).lower().startswith('e_'):
                        j += 1
                        break
                    j += 1

                i = j
                if entity_equal_flag:
                    true_positive_dict[label] += 1
            else:
                i += 1

    return dict(true_positive_dict)


def compute_prf_from_count_dict(true_positive_dict, true_dict, predict_dict):
    prf_dict = {}
    for label, true_count in true_dict.items():
        tp = true_positive_dict.get(label, 0)
        ap = predict_dict.get(label, 0)
        at = true_dict.get(label, 0)
        p = 0
        r = 0
        if ap != 0:
            p = tp / ap
        if at != 0:
            r = tp / at
        f = compute_f_value(p, r)
        prf_dict[label] = [round(float(item), 2) for item in (p * 100, r * 100, f * 100)]
    return prf_dict


def compute_f_value(p, r):
    f = 0
    if p != 0 and r != 0:
        f = 2 * p * r / (p + r)
    return f


def compute_confusion_matrix(true_data, predict_data, ner_labels):
    tag_map = ner_labels
    labels = tag_map.keys()
    labels_len = len(labels)
    label2id = {label: indx for indx, label in enumerate(labels)}

    confusion_matrix = np.zeros(shape=[labels_len, labels_len], dtype=np.int32)
    # confusion_matrix = pd.DataFrame(np.zeros(shape=[labels_len, labels_len], dtype=np.int32), columns=labels,
    #                                 index=labels)
    data_len = len(true_data)
    i = 0
    while i < data_len:
        tmp_true_item = str(true_data[i])
        tmp_predict_item = str(predict_data[i])
        if tmp_true_item.startswith('b_'):
            tmp_true_entity = []
            tmp_predict_entity = []
            tmp_true_entity.append(tmp_true_item)
            tmp_predict_entity.append(tmp_predict_item)
            j = i + 1
            while j < data_len:
                if str(true_data[j]).startswith('i_') or str(true_data[j]).startswith('e_'):
                    tmp_true_entity.append(true_data[j])
                    tmp_predict_entity.append(predict_data[j])
                    j += 1
                else:
                    if tmp_true_entity and tmp_predict_entity:
                        tmp_true_tag = get_label_from_ann(tmp_true_entity[0])
                        tmp_predict_tag = get_label_from_ann(tmp_predict_entity[0])
                        if is_entity_equal(tmp_true_entity, tmp_predict_entity):
                            confusion_matrix[label2id.get(tmp_true_tag)][label2id.get(tmp_predict_tag)] += 1
                        else:
                            processed_predict_label = count_labels_frequency(tmp_predict_entity)
                            if processed_predict_label is None:
                                confusion_matrix[label2id.get(tmp_true_tag)][label2id.get(tmp_predict_tag)] += 1
                            else:
                                for tmp_predict_tag_, freq in processed_predict_label.items():
                                    if tmp_predict_tag_ != tmp_true_tag:
                                        confusion_matrix[label2id.get(tmp_true_tag)][
                                            label2id.get(tmp_predict_tag)] += freq
                    tmp_true_entity = []
                    tmp_predict_entity = []
                    break
            if tmp_true_entity and tmp_predict_entity:
                tmp_true_tag = get_label_from_ann(tmp_true_entity[0])
                tmp_predict_tag = get_label_from_ann(tmp_predict_entity[0])
                if is_entity_equal(tmp_true_entity, tmp_predict_entity):
                    confusion_matrix[label2id.get(tmp_true_tag)][label2id.get(tmp_predict_tag)] += 1
                else:
                    processed_predict_label = count_labels_frequency(tmp_predict_entity)
                    if processed_predict_label is None:
                        confusion_matrix[label2id.get(tmp_true_tag)][label2id.get(tmp_predict_tag)] += 1
                    else:
                        for tmp_predict_tag_, freq in processed_predict_label.items():
                            if tmp_predict_tag_ != tmp_true_tag:
                                confusion_matrix[label2id.get(tmp_true_tag)][label2id.get(tmp_predict_tag)] += freq
            i = j
        else:
            tmp_true_tag = get_label_from_ann(tmp_true_item)
            tmp_predict_tag = get_label_from_ann(tmp_predict_item)
            # 将标签转为小写 add by han
            confusion_matrix[label2id.get(tmp_true_tag.lower())][label2id.get(tmp_predict_tag.lower())] += 1
            i += 1
    return confusion_matrix, label2id


def is_entity_equal(entity1, entity2):
    for label1, label2 in zip(entity1, entity2):
        if label1 != label2:
            return False
    return True


def get_label_from_ann(tag):
    label = 'o'
    if tag != 'o':
        label = tag[2:]
    return label


def process_tags(data):
    """
    删除 i e开头的，仅保留b s o
    :param data:
    :return:
    """
    processed_data = []
    for item in data:
        item = str(item).lower()
        if item.startswith('i_') or item.startswith('e_'):
            continue
        elif item.startswith('b_') or item.startswith('s_'):
            processed_data.append(get_label_from_ann(item))
        elif item == 'o':
            processed_data.append(item)
    return processed_data


class NerCfgData:
    def __init__(self):
        config_path = '{}/file_name_conf.cfg'.format(current_dir_name)
        config = configparser.ConfigParser(allow_no_value=True)
        config.read_file(codecs.open(config_path, "r", "utf-8"))
        self.ner_tags = config["tags"]

    def generate_tag_to_label(self):
        """
        获取配置中所有标签并标记id
        :return:
        """
        label2id = {}
        count = 0
        label2id['o'] = count
        for label in self.ner_tags:

            # t_b_l = 'b_{}'.format(label)
            # t_i_l = 'i_{}'.format(label)
            # t_e_l = 'e_{}'.format(label)
            # t_s_l = 's_{}'.format(label)
            t_b_l = 'B_{}'.format(label.upper())
            t_i_l = 'I_{}'.format(label.upper())
            t_e_l = 'E_{}'.format(label.upper())
            t_s_l = 'S_{}'.format(label.upper())
            label2id[t_b_l] = count + 1
            label2id[t_i_l] = count + 2
            label2id[t_e_l] = count + 3
            label2id[t_s_l] = count + 4
            count += 4
        return label2id


def get_tmp_file_name(dir_name):
    file_name = "{}/output.txt".format(dir_name)
    return file_name


def write_result(sent_arr, sent_predict_label_arr, label2tag, file_path, predict=False):
    file_handler = open(file_path, 'w')
    index = 0
    content = ""
    write_sent_flag = False
    for tmp_index in range(len(sent_arr)):
        tmp_sent = sent_arr[tmp_index]
        tmp_sent_word = tmp_sent[0]
        tmp_sent_pos = tmp_sent[1]
        tmp_sent_real_tag = tmp_sent[2]
        tmp_sent_predict_label = sent_predict_label_arr[tmp_index]
        write_sent_flag = False
        for tmp_row_index in range(len(tmp_sent_word)):
            tmp_word = tmp_sent_word[tmp_row_index]
            tmp_pos = tmp_sent_pos[tmp_row_index]
            tmp_real_tag = ''
            if not predict:
                tmp_real_tag = tmp_sent_real_tag[tmp_row_index]
            tmp_predict_label = tmp_sent_predict_label[tmp_row_index]
            tmp_predict_tag = label2tag[tmp_predict_label]
            content += "{}\t{}\t{}\t{}\n".format(tmp_word, tmp_pos, tmp_real_tag, tmp_predict_tag)
            index += 1
            write_sent_flag = True
        if write_sent_flag:
            content += "\n"
            index += 1
        if index > 1000:
            file_handler.write(content)
            content = ""
            index = 0
    if index > 0:
        file_handler.write(content)
    file_handler.close()


if __name__ == '__main__':
    import time

    t0 = time.time()
    # f = compute_prf_score('output2.txt')
    # print('old cost time is {}'.format(time.time() - t0))
    # print(f)
    # print('==================')
    t1 = time.time()
    f1 = compute_prf_score('./output/output.txt')
    print('new cost time is {}'.format(time.time() - t1))
    print(f1)
