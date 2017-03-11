import tensorflow as tf
import numpy as np
import math

import logging

log = logging.getLogger(__name__)

def round_up(a,b):
    return ((a + b - 1) // b) * b
def round_down(a,b):
    return (a // b) * b

class TrainingManager:
    def __init__(self, talks, batch_size, bptt_len):
        self.bptt_len = bptt_len
        self.talks = talks
        self.talks_start = np.zeros([len(self.talks)], dtype=np.int32)
        self.batch_size = batch_size

        # We subtract 1 because the final word can only be used for ground truth
        self.total_talk_len = sum(len(s) for s in self.talks)
        self.total_len = round_up(self.total_talk_len - 1, self.bptt_len) + 1
        self.len_of_wholes = round_down(self.total_talk_len - 1, self.bptt_len)

        if (self.total_talk_len - 1) % self.bptt_len < (self.bptt_len / 2):
            log.warning(("The length residue is small, {:d} whereas "
                         "bptt_len={:d}").format(self.total_talk_len %
                         self.bptt_len, self.bptt_len))

        self.training_data = np.zeros([batch_size, self.total_len], dtype=np.int32)
        seq_len = np.ones([batch_size], dtype=np.int32)
        self.sequence_length = seq_len * self.bptt_len
        self.partial_sequence_length = seq_len * ((self.total_talk_len - 1) %
                                                 self.bptt_len)
        self.new_training_permutation()
        self.epoch_steps = self.total_len // self.bptt_len

    def new_training_permutation(self):
        np.random.shuffle(self.talks)
        self.talks_start[0] = 0
        for i in range(1, len(self.talks)):
            # Negative rolls move the array to the left
            self.talks_start[i] = \
                self.talks_start[i-1] - len(self.talks[i-1])
        concat_talks = np.concatenate(self.talks)
        indices = np.random.choice(self.talks_start,
                                   size=self.batch_size % len(self.talks),
                                   replace=False)
        n_whole_talks = self.batch_size // len(self.talks)
        if n_whole_talks != 0:
            indices = np.concatenate([indices, np.tile(self.talks_start,
                                                       n_whole_talks)])
        assert indices.shape[0] == self.batch_size
        for batch_i, i in enumerate(indices):
            self.training_data[batch_i, :self.total_talk_len] = (
                np.roll(concat_talks, i))
        return self.training_data

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i+self.bptt_len+1 > self.total_len:
            raise StopIteration
        i = self.i
        self.i += self.bptt_len
        r = self.training_data[:, i:self.i+1]
        if i >= self.len_of_wholes:
            return r, self.partial_sequence_length
        return r, self.sequence_length

class TestManager:
    def new_training_permutation(self):
        pass

    def __init__(self, talks, batch_size, bptt_len):
        batch_size = min(batch_size, len(talks))
        def talks_concat(talks):
            "Greedily concatenate as many talks as necessary to have a batch"
            " of batch_size or less."
            ts = list([0, []] for _ in range(batch_size))
            for t in talks:
                min_len, min_len_i = ts[0][0], 0
                for i in range(batch_size):
                    if ts[i][0] < min_len:
                        min_len = ts[i][0]
                        min_len_i = i
                ts[min_len_i][0] += len(t)
                ts[min_len_i][1].append(t)
            talks = list(np.concatenate(t[1]) for t in ts)
            return talks

        talks = talks_concat(sorted(talks, key=lambda e: -len(e)))
        talks = sorted(talks, key=lambda e: -len(e))

        cutoffs = []
        talk_start = 0
        for i in range(len(talks)-1, -1, -1):
            while talk_start+1 < len(talks[i]):
                cutoffs.append(i+1)
                talk_start += bptt_len
        self.talk_arrays = []

        talk_start = 0
        for talk_start, cutoff in enumerate(cutoffs):
            talk_start *= bptt_len

            ts = np.zeros([cutoff, bptt_len+1], dtype=np.int32)
            tlens = np.zeros([cutoff], dtype=np.int32)
            for i in range(cutoff):
                n = tlens[i] = min(len(talks[i])-talk_start-1, bptt_len)
                ts[i,:n+1] = talks[i][talk_start:talk_start+n+1]
            self.talk_arrays.append((ts,tlens))

        self.epoch_steps = len(self.talk_arrays)

    def __iter__(self):
        return iter(self.talk_arrays)

class Model:
    input = None
    preds = None
    labels = None

    @staticmethod
    def number_to_label(i):
        a = np.zeros(shape=(8,), dtype=np.float32)
        a[i] = 1
        return a
    def build_tf(self):
        raise NotImplementedError("Build base class Model")

class LSTMBase(Model):
    def __init__(self, num_units, num_layers, num_chars, embedding_trainable, training_keep_prob,
                 bptt_length, softmax_samples, batch_size):
        self.training_keep_prob = training_keep_prob
        self.keep_prob = tf.placeholder(tf.float32, shape=[])
        self.input = tf.placeholder(tf.int32, shape=[None, bptt_length+1],
                                    name="input")
        self.input_infer = tf.slice(self.input, [0, 0], [tf.shape(self.input)[0],
                                                         bptt_length])
        self.ground_truth = tf.slice(self.input, [0, 1], [tf.shape(self.input)[0],
                                                          bptt_length])
        self.max_batch_size = -1
        self.default_batch_size = batch_size

        self.sequence_length = tf.placeholder(tf.int32, shape=[None],
                                              name="sequence_length")
        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[])

        self.embeddings = tf.get_variable("embeddings", dtype=tf.float32,
                                          shape=[num_chars, num_units],
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          trainable=embedding_trainable)
        self.embedded = tf.nn.embedding_lookup(self.embeddings, self.input_infer)

        one_cell = lambda: tf.contrib.rnn.LayerNormBasicLSTMCell(num_units,
            dropout_keep_prob=self.keep_prob)
        cells = list(one_cell() for _ in range(num_layers))
        self.cell = tf.contrib.rnn.MultiRNNCell(cells)
        initial_state_tf = self.create_initial_state_placeholder()
        self.rnn_outputs, next_state = tf.nn.dynamic_rnn(
            self.cell,
            inputs=self.embedded,
            sequence_length=self.sequence_length,
            initial_state=initial_state_tf,
            dtype=tf.float32)
        self.create_next_state_tensor_list(next_state)

        with tf.variable_scope("sequence_mask"):
            mask = tf.sequence_mask(self.sequence_length, bptt_length)
            masked_rnn_outs = tf.boolean_mask(self.rnn_outputs, mask)
            masked_truth = tf.boolean_mask(self.ground_truth, mask)

        with tf.variable_scope("softmax_layer"):
            self.W = tf.get_variable("W", dtype=tf.float32,
                                     shape=[num_chars, self.cell.output_size],
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)
            self.b = tf.get_variable("b", dtype=tf.float32,
                                     shape=[num_chars],
                                     initializer=tf.constant_initializer(0.1),
                                     trainable=True)
        logits = tf.matmul(masked_rnn_outs, tf.transpose(self.W)) + self.b
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=masked_truth)
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")
        self.entropy_sum = cross_entropy_sum / math.log(2)
        self.prediction_sample = tf.squeeze(tf.multinomial(logits, 1))
        self.n_samples = tf.to_float(tf.shape(logits)[0])
        self.prediction_max = tf.argmax(logits, axis=1)

        if self.LOSS_FUNCTION == 'sampled_softmax':
            raise Exception('stop using that')
            loss_ = tf.nn.sampled_softmax_loss(self.W, self.b,
                        inputs=masked_rnn_outs,
                        labels=tf.expand_dims(masked_truth, 1),
                        num_sampled=softmax_samples,
                        num_classes=embedding_matrix.shape[0],
                        num_true=1)
            self.loss = tf.reduce_mean(loss_)
        elif self.LOSS_FUNCTION == 'cross_entropy':
            self.loss = cross_entropy_sum / self.n_samples
        else:
            raise ValueError("self.LOSS_FUNCTION cannot be "
                             " {:s}".format(self.LOSS_FUNCTION))

    def train_step(self, Optimizer):
        return Optimizer(learning_rate=self.learning_rate_ph).minimize(self.loss)

    def create_initial_state_placeholder(self):
        self.initial_state = []
        initial_state_tf = []
        for j, sz in enumerate(self.cell.state_size):
            if isinstance(sz, int):
                sizes = (sz,)
            else:
                sizes = tuple(sz)
            l = []
            for i, s in enumerate(sizes):
                ph = tf.placeholder(tf.float32, shape=[None, s],
                    name='initial_state_{:d}_{:d}'.format(j, i))
                self.initial_state.append(ph)
                l.append(ph)
            if len(l) > 1:
                initial_state_tf.append(tf.contrib.rnn.LSTMStateTuple(*l))
            else:
                initial_state_tf.append(l[0])
        return tuple(initial_state_tf)

    def create_next_state_tensor_list(self, next_state):
        self.next_state = []
        for ns in next_state:
            if isinstance(ns, tuple):
                self.next_state += ns
            else:
                self.next_state.append(ns)

    def new_epoch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.default_batch_size
        if batch_size > self.max_batch_size:
            log.warning("Increased batch size to {:d}".format(batch_size))
            self.initial_state_value = []
            for s in self.initial_state:
                self.initial_state_value.append(np.zeros(
                    [batch_size, s.get_shape()[1]], dtype=np.float32))
            self.max_batch_size = batch_size
        self.state = list(s[:batch_size] for s in self.initial_state_value)

    def training_feed_dict(self, training_example, learning_rate):
        l = len(training_example[1])
        assert self.state[0].shape[0] >= l, (
               "Batch size increased from {:d} to {:d}".format(
                   self.state[0].shape[0], l))
        d = {self.input: training_example[0],
                self.sequence_length: training_example[1],
                self.learning_rate_ph: learning_rate,
                self.keep_prob: self.training_keep_prob}
        for i, s in enumerate(self.initial_state):
            d[s] = self.state[i][:l]
        return d

    def test_feed_dict(self, test_example):
        l = len(test_example[1])
        assert self.state[0].shape[0] >= l, (
               "Batch size increased from {:d} to {:d}".format(
                   self.state[0].shape[0], l))
        d = {self.input: test_example[0],
                self.sequence_length: test_example[1],
                self.keep_prob: 1.0}
        for i, s in enumerate(self.initial_state):
            d[s] = self.state[i][:l]
        return d

    def compute_entropy(self, sess, test_mgr):
        total_entropy = 0.0
        total_entropy_normalise = 0
        self.new_epoch()
        for example in test_mgr:
            res = sess.run([self.entropy_sum, self.n_samples] + self.next_state,
                self.test_feed_dict(example))
            total_entropy += res[0]
            total_entropy_normalise += res[1]
            self.state = res[2:]
        return total_entropy/total_entropy_normalise

class SampledSoftmaxLSTM(LSTMBase):
    LOSS_FUNCTION = 'sampled_softmax'

class SoftmaxLSTM(LSTMBase):
    LOSS_FUNCTION = 'cross_entropy'
