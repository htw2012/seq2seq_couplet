#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.platform import gfile

import data_tools
import seq2seq_model

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 30, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 5000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "data/",
                           "Data directory")
tf.app.flags.DEFINE_string("train_dir", "train_result/",
                           "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(17, 4), (33, 8), (65, 16), (132, 33)]


def read_data(source_path, target_path, max_size=None):
    """
    读取source_path和target_path端数据
    :param source_path:
    :param target_path:
    :param max_size:
    :return:
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:

            # 单行读取
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]

                # target默认追加eos
                target_ids.append(data_tools.EOS_ID)

                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                # 单行读取
                source, target = source_file.readline(), target_file.readline()
    return data_set


def create_model(session, forward_only):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.en_vocab_size,
        FLAGS.fr_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        forward_only=forward_only,
        dtype=dtype)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables()) # deprecated
        # session.run(tf.global_variables_initializer) #不能用了
    return model


def train():
    # Preparing train/dev
    print("Preparing train/dev data in %s" % FLAGS.data_dir)
    en_train, fr_train, en_dev, fr_dev, _, _ = data_tools.prepare_wmt_data(
        FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocab_size)

    # Assume that you have 8GB of GPU memory and want to allocate ~4GB:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)

        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)."
              % FLAGS.max_train_data_size)

        # 按桶读取,训练集和测试集
        dev_set = read_data(en_dev, fr_dev)
        train_set = read_data(en_train, fr_train, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "p2w.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                        "inf")
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


def decode():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        en_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.en" % FLAGS.en_vocab_size)
        fr_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.fr" % FLAGS.fr_vocab_size)
        src_vocab, _ = data_tools.initialize_vocabulary(en_vocab_path)
        _, rev_fr_vocab = data_tools.initialize_vocabulary(fr_vocab_path)

        # 读取测试源端和目标端句子
        dev_src_texts, dev_tgt_texts = data_tools.do_test(FLAGS.data_dir)

        outputPath = os.path.join(FLAGS.data_dir, "dev_data.result")
        correct = 0
        with gfile.GFile(outputPath, mode="w") as outputfile:
            for i in xrange(len(dev_src_texts)):
                src_text = dev_src_texts[i]

                # 获得原始输入句子的索引id
                token_ids = data_tools.sentence_to_token_ids(tf.compat.as_bytes(src_text), src_vocab)

                # 获得句子分配的桶位置
                bucket_id = min([b for b in xrange(len(_buckets))
                                 if _buckets[b][0] > len(token_ids)])

                # 根据相应的桶和输入信息,获得seq2seq的编码器的输入,解码器的输入和目标权值
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    {bucket_id: [(token_ids, [])]}, bucket_id)

                # Get output logits for the sentence.
                _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                # This is a greedy decoder - outputs are just argmaxes of output_logits.
                outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                # If there is an EOS symbol in outputs, cut them at that point.
                if data_tools.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(data_tools.EOS_ID)]
                # Print out French sentence corresponding to outputs.
                out = " ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs])
                out = "M " + out + "\n"
                if out == dev_tgt_texts[i]:
                    correct += 1
                outputfile.write('intput: %s' % src_text)
                outputfile.write('output: %s' % out)
                outputfile.write('target: %s' % dev_tgt_texts[i])
                outputfile.write('\n')
            precision = correct / len(dev_tgt_texts)
            outputfile.write('precision = %.3f' % precision)

def decode_once2(sess, model, input):

    # model = Singleton.get_instance(sess)
    model.batch_size = 1  # We decode one sentence at a time.
    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.en" % FLAGS.en_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.fr" % FLAGS.fr_vocab_size)
    en_vocab, _ = data_tools.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = data_tools.initialize_vocabulary(fr_vocab_path)

    sentence = input
    # Get token-ids for the input sentence.
    token_ids = data_tools.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
    # print("token:%s"%token_ids)
    # print("token-len:%s"%len(token_ids))
    # Which bucket does it belong to?
    l = [b for b in xrange(len(_buckets))
                     if _buckets[b][0] > len(token_ids)]
    # print("l", l)
    if l:
        bucket_id = min(l)
    else:
        bucket_id = 3

    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id)
    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

    # If there is an EOS symbol in outputs, cut them at that point.
    if data_tools.EOS_ID in outputs:
      outputs = outputs[:outputs.index(data_tools.EOS_ID)]
    # Print out French sentence corresponding to outputs.
    outputresult = " ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs])

    return outputresult

def main(_):
    decode()


if __name__ == "__main__":
    tf.app.run()
