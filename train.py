#!/usr/bin/env python3

import tensorflow as tf
import logging
import math
import os
from collections import defaultdict
import pickle
from parse import TextLoader
import model
import numpy as np
from model import SoftmaxLSTM

flags = tf.app.flags

flags.DEFINE_boolean('load_latest', False, 'Whether to load the last model')
flags.DEFINE_string('load_latest_from', '', 'Folder to load the model from')
flags.DEFINE_string('load_from', '', 'File to load the model from')
flags.DEFINE_boolean('log_to_stdout', False, 'Whether to output the python log '
                     'to stdout, or to a file')
flags.DEFINE_string('input_type', 'character', '[word,character]')
flags.DEFINE_string('out_file', 'out', 'the file to output test scores / sample phrases to')
flags.DEFINE_string('command', 'train', 'What to do [train, validation, test, generate_sample]')
flags.DEFINE_boolean('embedding_trainable', True, 'Train embeddings?')
flags.DEFINE_string('log_dir', './logs/', 'Base directory for logs')
flags.DEFINE_string('model_dir', './model/', 'Base directory for model')
flags.DEFINE_integer('batch_size', 50, 'batch size for training')
flags.DEFINE_integer('max_epochs', 1000, 'maximum training epochs')
flags.DEFINE_integer('hidden_units', 512, 'Number of hidden units per LSTM layer')
flags.DEFINE_integer('hidden_layers', 2, 'Number of hidden LSTM layers')
flags.DEFINE_integer('bptt_len', 100, 'Number of tokens to backpropagate through')
flags.DEFINE_string('nonlinearity', 'tanh', 'Nonlinearity of the hidden layer')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate for ADAM')
flags.DEFINE_float('learning_rate_decay', 0.99, 'Exponential decay for learning rate')
flags.DEFINE_float('min_learning_rate', 0.005, 'Minimum learning rate')
flags.DEFINE_float('dropout', 0.5, 'probability of keeping a neuron on')
flags.DEFINE_integer('softmax_samples', 10, 'Number of samples for the sampled softmax loss')
flags.DEFINE_string('log_level', 'DEBUG', 'logging level')

FILENAME_FLAGS = ['batch_size', 'learning_rate', 'learning_rate_decay',
                  'hidden_units', 'hidden_layers', 'bptt_len', 'dropout']

FLAGS = flags.FLAGS

log = logging.getLogger(__name__)

def get_relevant_directories():
    # Give the model a nice name in TensorBoard
    current_flags = []
    for flag in FILENAME_FLAGS:
        current_flags.append('{}={}'.format(flag, getattr(FLAGS, flag)))
    _log_dir = FLAGS.log_dir = os.path.join(FLAGS.log_dir, *current_flags)
    _model_dir = FLAGS.model_dir = os.path.join(FLAGS.model_dir, *current_flags)
    i=0
    while os.path.exists(FLAGS.log_dir):
        i += 1
        FLAGS.log_dir=('{}/{}'.format(_log_dir, i))
    FLAGS.log_dir=('{}/{}'.format(_log_dir, i))
    log_file = os.path.join(FLAGS.log_dir, 'console_log.txt')
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    basicConfigKwargs = {'level': getattr(logging, FLAGS.log_level.upper()),
                         'format': '%(asctime)s %(name)s %(message)s'}
    if not FLAGS.log_to_stdout:
        basicConfigKwargs['filename'] = log_file
    logging.basicConfig(**basicConfigKwargs)
    save_model_file=('{}/{}/ckpt'.format(_model_dir, i))
    save_model_dir=('{}/{}'.format(_model_dir, i))
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    if FLAGS.load_latest or FLAGS.load_latest_from:
        if FLAGS.load_latest:
            load_dir=('{}/{}'.format(_model_dir, i-1))
        else:
            load_dir=FLAGS.load_latest_from
        load_file = tf.train.latest_checkpoint(load_dir)
        if load_file is None:
            log.error("No checkpoint found!")
            exit(1)
    elif FLAGS.load_from:
        load_file = FLAGS.load_from
    else:
        load_file = None
    return FLAGS.log_dir, save_model_file, load_file

def main(_):
    log_dir, save_model_file, load_file = get_relevant_directories()
    tl = TextLoader("save")
    word_index = tl.vocab
    word_index_f = lambda c: word_index[c]
    reverse_word_index = tl.reverse_vocab
    reverse_word_index_f = lambda c: reverse_word_index[c]
    for k, v in enumerate(reverse_word_index):
        if len(v) > 1:
            reverse_word_index[k] = '('+v+')'
    del tl
    with open('save/parsed/sequences.pkl', 'rb') as f:
        data = pickle.load(f)
    np.random.shuffle(data)
    training = model.TestManager(data[:-10], FLAGS.batch_size, FLAGS.bptt_len)
    validation = model.TestManager(data[-10:-5], FLAGS.batch_size, FLAGS.bptt_len)
    test = model.TestManager(data[-5:], FLAGS.batch_size, FLAGS.bptt_len)
    del data

    log.info("Building model...")
    sess = tf.Session()
    m = SoftmaxLSTM(num_units=FLAGS.hidden_units,
                    num_layers=FLAGS.hidden_layers,
                    num_chars=len(word_index),
                    embedding_trainable=FLAGS.embedding_trainable,
                    training_keep_prob=FLAGS.dropout,
                    bptt_length=FLAGS.bptt_len,
                    softmax_samples=FLAGS.softmax_samples,
                    batch_size=FLAGS.batch_size)
    train_step = m.train_step(tf.train.AdamOptimizer)

    # Model checkpoints and graphs
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(tf.global_variables_initializer())
    if load_file:
        saver.restore(sess, load_file)
        log.info("Loaded model from file %s" % load_file)
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    training_summary = tf.summary.scalar('training/loss', m.loss)
    perplexity_ph = tf.placeholder(tf.float32, shape=[])
    training_perp_summary = tf.summary.scalar('training/entropy', perplexity_ph)
    validation_summary = tf.summary.scalar('validation/entropy', perplexity_ph)

    if FLAGS.command == 'train':
        learning_rate = FLAGS.learning_rate
        log.info("Each epoch has {:d} steps.".format(training.epoch_steps))
        for epoch in range(1, FLAGS.max_epochs+1):
            log.info("Training epoch %d..." % epoch)
            training.new_training_permutation()
            m.new_epoch()
            for i, training_example in enumerate(training):
                summary_t = (epoch-1) * training.epoch_steps + i
                feed_dict = m.training_feed_dict(training_example, learning_rate)
                if i % 10 == 9:
                    log.info("Running example {:d}".format(i+1))
                result = sess.run([training_summary, train_step, m.entropy_sum, m.n_samples] +
                                  m.next_state, feed_dict)
                summary_writer.add_summary(result[0], summary_t)
                training_ent = result[2]/result[3]
                summary, = sess.run([training_perp_summary], {perplexity_ph: training_ent})
                summary_writer.add_summary(summary, summary_t)
                if i % 1000 == 999 or i == training.epoch_steps-1:
                    def compute_perplexity(test_mgr, summary_op):
                        perplexity = m.compute_entropy(sess, test_mgr)
                        summary, = sess.run([summary_op],
                                            {perplexity_ph: perplexity})
                        summary_writer.add_summary(summary, summary_t)
                        return perplexity

                    perplexity = compute_perplexity(validation, validation_summary)
                    log.info("Validation entropy is {:.4f}".format(perplexity))
                    if math.isnan(perplexity) or math.isinf(perplexity):
                        import pdb
                        pdb.set_trace()

                    log.info("Generating 10 phrases:")
                    generated_seqs = m.generate_sequences(sess,
                        [[word_index['START']]]*10,
                        20 if FLAGS.input_type == "word" else 100)
                    for phrase in generated_seqs:
                        log.info("".join(map(reverse_word_index_f, phrase)))
                    save_path = saver.save(sess, save_model_file, global_step=summary_t)
                    log.info("Model saved in file: {:s}, validation entropy {:.4f}"
                             .format(save_path, perplexity))
                    learning_rate = max(learning_rate*FLAGS.learning_rate_decay,
                                        FLAGS.min_learning_rate)
                    log.info("New learning rate is {:f}".format(learning_rate))
                m.state = result[4:]
    elif FLAGS.command == 'validation':
        m.new_epoch()
        ent = m.compute_entropy(sess, validation)
        print("Validation entropy:", ent)
    elif FLAGS.command == 'test':
        m.new_epoch()
        m.compute_entropy(sess, test)
        print("Test entropy:", ent)
    elif FLAGS.command == 'generate_sample':
        log.info("Generating sample...")
        n_generate = 20
        m.new_epoch(n_generate)
        seed = list(map(word_index_f, ["long-delay", "start-sayer-nick"] + list("funcoooo") + ["utter"] + "hi, how are you?"+ ["end-event"]))
        generated_seqs = m.generate_sequences(sess, [seed]*n_generate, 50)
        for phrase in generated_seqs:
            print("".join(map(reverse_word_index_f, phrase)))
    sess.close()

if __name__ == '__main__':
    tf.app.run()

