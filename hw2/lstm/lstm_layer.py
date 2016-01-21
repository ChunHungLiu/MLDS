import theano
import theano.tensor as T
import numpy as np

import func

def Sigmoid(z):
    return 1 / (1 + T.exp(-z))

class LSTMLayer:
    def __init__(self, batch_size, n_inputs, n_outputs, x_seq, shared_params, act_type = "ReLU"):
        assert len(shared_params)  == 8
        W, Wi, Wf, Wo, b, bi, bf, bo = shared_params

        assert W.get_value().shape == ( n_inputs + 2 * n_outputs, n_outputs )

        c_0  = theano.shared( value = np.zeros((batch_size,n_outputs)).astype('float32') )
        y_0  = theano.shared( value = np.zeros((batch_size,n_outputs)).astype('float32') )

        def step( x_t, c_tm1, y_tm1 ):
            con = T.concatenate([ x_t, c_tm1, y_tm1 ], axis = 1)

            z  = T.dot( con, W)  + b.dimshuffle('x',0)
            zi = T.dot( con, Wi) + bi.dimshuffle('x',0)
            zf = T.dot( con, Wf) + bf.dimshuffle('x',0)
            zo = T.dot( con, Wo) + bo.dimshuffle('x',0)

            c = Sigmoid(z) * Sigmoid(zi) + c_tm1 * Sigmoid(zf)
            y = Sigmoid(c) * Sigmoid(zo)

            c_ret = func.activation_func(c,act_type)
            y_ret = func.activation_func(y,act_type)

            return c_ret, y_ret

        [ c_seq, y_seq ], _ = theano.scan( fn = step,
                                         sequences = x_seq,
                                         outputs_info = [ c_0, y_0 ] )
        self.y_seq = y_seq
        self.prop = theano.function( inputs = [ x_seq ],
                                     outputs = y_seq )

class LSTM_last_layer:
    def __init__(self , Wo , bo , a_seq, act_type = "RELU" ):
        self.y_seq = T.dot( a_seq , Wo ) + bo.dimshuffle( 'x', 0 )
        #self.y_seq = func.activation_func( T.dot(a_seq , Wo) + bo.dimshuffle('x',0), act_type )
