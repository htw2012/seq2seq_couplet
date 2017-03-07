#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;/)(])")
_DIGIT_RE = re.compile(br"\d")


def basic_tokenizer(sentence):
    # words = []
    # for space_separated_fragment in sentence.strip().split():
    #     if space_separated_fragment != 'M' and space_separated_fragment != 'E':
    #         words.extend(_WORD_SPLIT.split(space_separated_fragment))
    sentence = sentence.strip('\n')
    sentence = sentence.decode('utf-8')
    ret = [w for w in sentence]
    return ret


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
    """Create vocabulary file (if it does not exist yet) from data file.
    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.
    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.
    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].
    Args:
      vocabulary_path: path to the file containing the vocabulary.
    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).
    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip().decode('utf-8') for line in rev_vocab] # 这里做了转换为UTF-8 todo
        # rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.
    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    Returns:
      a list of integers, the token-ids for the sentence.
    """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    # return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]
    # print("words:%s"%(words))
    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.
    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.
    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                      normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_wmt_data(data_dir, en_vocabulary_size, fr_vocabulary_size, tokenizer=None):
    """Get WMT data into data_dir, create vocabularies and tokenize data.
    Args:
      data_dir: directory in which the data sets will be stored.
      en_vocabulary_size: size of the English vocabulary to create and use.
      fr_vocabulary_size: size of the French vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for English training data-set,
        (2) path to the token-ids for French training data-set,
        (3) path to the token-ids for English development data-set,
        (4) path to the token-ids for French development data-set,
        (5) path to the English vocabulary file,
        (6) path to the French vocabulary file.
    """
    # 英文到法文-->音到字
    en_data_path = os.path.join(data_dir, "first.txt")
    fr_data_path = os.path.join(data_dir, "next.txt")

    train_path = os.path.join(data_dir, "train_data")
    dev_path = os.path.join(data_dir, "dev_data")
    create_train_dev_data(en_data_path, fr_data_path, train_path, dev_path)

    # Create vocabularies of the appropriate sizes.
    fr_vocab_path = os.path.join(data_dir, "vocab%d.fr" % fr_vocabulary_size)
    en_vocab_path = os.path.join(data_dir, "vocab%d.en" % en_vocabulary_size)
    create_vocabulary(fr_vocab_path, train_path + ".fr", fr_vocabulary_size, tokenizer)
    create_vocabulary(en_vocab_path, train_path + ".en", en_vocabulary_size, tokenizer)

    # Create token ids for the training data.
    fr_train_ids_path = train_path + (".ids%d.fr" % fr_vocabulary_size)
    en_train_ids_path = train_path + (".ids%d.en" % en_vocabulary_size)
    data_to_token_ids(train_path + ".fr", fr_train_ids_path, fr_vocab_path, tokenizer)
    data_to_token_ids(train_path + ".en", en_train_ids_path, en_vocab_path, tokenizer)

    # Create token ids for the development data.
    fr_dev_ids_path = dev_path + (".ids%d.fr" % fr_vocabulary_size)
    en_dev_ids_path = dev_path + (".ids%d.en" % en_vocabulary_size)
    data_to_token_ids(dev_path + ".fr", fr_dev_ids_path, fr_vocab_path, tokenizer)
    data_to_token_ids(dev_path + ".en", en_dev_ids_path, en_vocab_path, tokenizer)

    return (en_train_ids_path, fr_train_ids_path,
            en_dev_ids_path, fr_dev_ids_path,
            en_vocab_path, fr_vocab_path)


def create_train_dev_data(en_data_path, fr_data_path, train_path, dev_path):
    if gfile.Exists(train_path + ".fr"):
        print("has already create_train_dev_data....")
        return

    if gfile.Exists(en_data_path) and gfile.Exists(fr_data_path):
        print("Get data src from %s,target from %s" % (en_data_path, fr_data_path))
        with gfile.GFile(en_data_path, mode="rb") as en_data_file, \
                gfile.GFile(fr_data_path, mode="rb") as fr_data_file:
            with gfile.GFile(train_path + ".fr", mode="w") as train_fr, \
                    gfile.GFile(train_path + ".en", mode="w") as train_en, \
                    gfile.GFile(dev_path + ".fr", mode="w") as dev_fr, \
                    gfile.GFile(dev_path + ".en", mode="w") as dev_en:
                counter = 0
                fr = train_fr
                en = train_en
                for en_line, fr_line in zip(en_data_file,fr_data_file):
                    counter += 1
                    if counter % 1000 == 0:
                        print("  get line %d" % counter)

                    ### 处理en和fr句子
                    if counter > 9588:# 4:1划分训练集和测试集
                        fr = dev_fr
                        en = dev_en

                    #### 长度约束,不约束,可能长度有问题,需要encoder
                    if len(en_line)>130 or len(fr_line.decode('utf-8')) > 32:
                        continue

                    ### 拼音长度控制在130个 TODO
                    ### 汉字长度控制在32个
                    en.write(en_line) # 是否自己换行
                    fr.write(fr_line)

def find_longest(data_path):
    count = 0
    if gfile.Exists(data_path):
        print("Get data from %s" % data_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            for line in data_file:
                words = basic_tokenizer(line)
                if count < len(words):
                    count = len(words)
                    print(line)
    return count

def get_max_min():
    source_path = "data/train_data.ids30.en"
    target_path = "data/train_data.ids5000.fr"
    max_src = 0
    min_src = 100

    max_tgt = 0
    min_tgt = 100
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            # 单行读取
            source, target = source_file.readline(), target_file.readline()
            while source and target:
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(EOS_ID)

                # target默认追加eos
                src_size = len(source_ids)
                tgt_size = len(target_ids)

                max_src = max(max_src, src_size)
                min_src = min(min_src, src_size)

                max_tgt = max(max_tgt, tgt_size)
                min_tgt = min(min_tgt, tgt_size)
                # 单行读取
                source, target = source_file.readline(), target_file.readline()

            print("max_src",max_src, "min_src",min_src)
            print("max_tgt",max_tgt, "min_tgt",min_tgt)

def do_test(data_dir):
    dev_src_texts = []
    dev_tgt_texts = []
    data_path = os.path.join(data_dir, "dev_data.en")
    if gfile.Exists(data_path):
        print("Get data from %s" % data_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            for line in data_file:
                dev_src_texts.append(line)

    data_path_org = os.path.join(data_dir, "dev_data.fr")
    if gfile.Exists(data_path_org):
        print("Get data from %s" % data_path_org)
        with gfile.GFile(data_path_org, mode="rb") as data_org_file:
            for line in data_org_file:
                dev_tgt_texts.append(line)
    return dev_src_texts, dev_tgt_texts


def main():
    data_dir = "data/"
    en_vocab_size = 3465
    fr_vocab_size = 3465
    en_train, fr_train, en_dev, fr_dev, _, _ = prepare_wmt_data(data_dir, en_vocab_size, fr_vocab_size)

    # line = "解放军强渡渭河"
    # # token_test(line)
    # #
    # vocab, _ = initialize_vocabulary("data/vocab5000.fr")
    # print("vocab:%s"%(vocab))
    # token_ids = sentence_to_token_ids(line, vocab, None, True)
    # print("ids:%s"%(token_ids))
    #
    # line = "jiefangjunqiangduweihe"
    # token_test(line)
    # vocab, _ = initialize_vocabulary("data/vocab30.en")
    # print("vocab:%s"%(vocab))
    # token_ids = sentence_to_token_ids(line, vocab, None, True)
    # print("ids:%s"%(token_ids))

    # source_path = "data/dev_data.ids30.en"
    # target_path = "data/dev_data.ids5000.fr"
    # with tf.gfile.GFile(source_path, mode="r") as source_file:
    #     with tf.gfile.GFile(target_path, mode="r") as target_file:
    #
    #         # 单行读取
    #         source, target = source_file.readline(), target_file.readline()
    #         counter = 0
    #         while source and target and (counter < 30):
    #             counter += 1
    #             source_ids = [int(x) for x in source.split()]
    #             target_ids = [int(x) for x in target.split()]
    #
    #             # target默认追加eos
    #             target_ids.append(EOS_ID)
    #             print("source_id", source_ids,"source_text", source)
    #             print("target_ids", target_ids,"target_text", target)
    #             print("source_ids-len:%s,target_ids:%s"%(len(source_ids), len(target_ids)))
    #             # 单行读取
    #             source, target = source_file.readline(), target_file.readline()
    # get_max_min()

def token_test(line):
    ret = basic_tokenizer(line)
    print("ret:%s" % (ret))
    print("len", len(ret))


if __name__ == "__main__":
    main()
