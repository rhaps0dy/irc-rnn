import numpy as np
import theano
import theano.tensor as T
import lasagne

from parse import TextLoader

class OneHotLayer(lasagne.layers.Layer):
    def __init__(self, incoming, n_states, **kwargs):
        """
        incoming: input layer to this one
        n_states: length of the one-hot vectors
        """
        super().__init__(incoming, **kwargs)
        self.eye = T.eye(n_states)

    def get_output_shape_for(self, input_shape):
        return tuple(list(input_shape) + [self.eye.shape[0]])

    def get_output_for(self, input, **kwargs):
        return self.eye[input]

tl = TextLoader("save")
tl.load()
tl.voca

l_in_enc = lasagne.layers.InputLayer(shape=(None, None, 2))
l_in_dec = lasagne.layers.InputLayer(shape=(None, None, 2))
l_one_hot_enc = OneHotLayer(l_in_enc, tl.n_chars, name='one_hot_enc')
l_enc = lasagne.layers.GRULayer(l_one_hot_enc, num_units=tl.n_chars, name='encoder', only_return_final=True)
l_one_hot_dec = OneHotLayer(l_in_dec, tl.n_chars, name='one_hot_dec')
l_dec = lasagne.layers.GRULayer(l_one_hot_dec, num_units=tl.n_chars, hid_init=l_enc, name='decoder')
# Change shape: (batch_size, decode_len, one_hot_size) -> (batch_size*decode_len, one_hot_size)
l_reshape = lasagne.layers.ReshapeLayer(l_dec, (-1, [2]))
l_softmax = lasagne.layers.DenseLayer(l_reshape, num_units=tl.n_chars,
                                      nonlinearity=lasagne.nonlinearities.softmax,
                                      name='softmax_output')
