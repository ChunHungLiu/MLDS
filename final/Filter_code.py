
class Matched_filter_opt(Layer):
    '''
        previous layer is the filter 
        Data_tobe_masked is the image 
    '''
    def __init__(self, data_dim = None ,input_dim = None ,**kwargs):
        self.input_dim = input_dim
        self.data_dim = data_dim 
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Matched_filter_opt, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]
        self.output_dim = self.data_dim 
        self.input = T.matrix()

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)
        #return (self.input_shape[0], self.input_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = (X.T[0:self.data_dim]).T * ( (X.T[self.data_dim:2*self.data_dim]).T )
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "input_dim": self.input_dim}
        base_config = super(Matched_filter_opt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


