'''
EJ sharing weight 
'''

class beShared_LSTM(Recurrent):
    def __init__(self, shared_layer , 
                 #output_dim,
                 #init='glorot_uniform', inner_init='orthogonal',
                 #forget_bias_init='one', activation='tanh',
                 #inner_activation='hard_sigmoid',
                 **kwargs):
        self.output_dim = shared_layer.output_dim
        self.init = shared_layer.init
        self.inner_init = shared_layer.inner_init 
        self.forget_bias_init = shared_layer.forget_bias_init 
        self.activation = shared_layer.activation
        self.inner_activation = shared_layer.inner_activation
        self.shared_layer = shared_layer
        super(beShared_LSTM, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.input = K.placeholder(input_shape)

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (output_dim)
            self.states = [None, None]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def step(self, x, states):
        assert len(states) == 2
        h_tm1 = states[0]
        c_tm1 = states[1]
        
        x_i = K.dot(x, self.shared_layer.W_i) + self.shared_layer.b_i
        x_f = K.dot(x, self.shared_layer.W_f) + self.shared_layer.b_f
        x_c = K.dot(x, self.shared_layer.W_c) + self.shared_layer.b_c
        x_o = K.dot(x, self.shared_layer.W_o) + self.shared_layer.b_o

        i = self.inner_activation(x_i + K.dot(h_tm1, self.shared_layer.U_i))
        f = self.inner_activation(x_f + K.dot(h_tm1, self.shared_layer.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.shared_layer.U_c))
        o = self.inner_activation(x_o + K.dot(h_tm1, self.shared_layer.U_o))
        h = o * self.activation(c)
        return h, [h, c]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(beShared_LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

   
