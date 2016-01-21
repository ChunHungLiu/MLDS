from itertools import izip

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

nprng = np.random.RandomState(1234)
rng = RandomStreams(nprng.randint(2 ** 30))

class DA:
    def __init__(self, n_hid, W, b_h, l_rate=0.05, mu=0.9, d_prob=0.1):
        self.W = W

        l_rate = T.scalar(dtype='float32')
        mu  = T.scalar(dtype='float32')
        d_prob = T.scalar(dtype='float32')
        
        #self.prev_grad = theano.shared(0., dtype=theano.config.floatX)
        n_vis = W.get_value().shape[1]
        assert n_hid == W.get_value().shape[0]
        data = T.matrix(dtype='float32')
        init = 0.5*np.sqrt(n_vis)
        '''
        b_v = theano.shared(
            np.random.uniform(-init,init,size=(n_hid)) )
        '''
        b_v = theano.shared(np.asarray(
            np.random.uniform(-init,init,size=(n_vis)),'float32' ))
        
        # input neurons are dropped by a prob.		
        drop_data = rng.binomial(size=data.shape, n=1, p=1-d_prob,
                                 dtype='float32') * data 
        hid = 1/(1+T.exp( -1*(T.dot(W, drop_data)+b_h.dimshuffle(0,'x')) ))
        res = 1/(1+T.exp( -1*(T.dot(W.T, hid)+b_v.dimshuffle(0,'x')) ))

        '''
        # apply softmax to output layer.
        z_res = T.dot(W.T, hid) + b_v.dimshuffle(0,'x')
        exp_sum = T.sum(T.exp(z_res), axis=0)
        softmax = T.exp(z_res) / exp_sum.dimshuffle('x',0)
        #softmax = T.clip(res, 1e-7, 1-1e-7)
        #self.debug = theano.function([data],softmax)
		'''

        params = [W, b_h]
        cost = T.sum( (data-res)**2 ) / T.cast(data.shape[1],'float32')
        #cost = -T.sum( data*T.log(softmax+1e-7) ) / T.cast(data.shape[1],'float32')
        grads = T.grad(cost, params)
        prev_updates = []
        for param in params:
            prev_updates.append( theano.shared(
                np.asarray(
                    np.zeros(param.get_value().shape),
                'float32' )
            ))

        def update(params, grads):
            parameter_updates = [ (p, p - l_rate*g + mu*v)  \
                for p,g,v in izip(params,grads,prev_updates) ]
            parameter_updates += [ (v, mu*v) \
                for v in prev_updates ]
            return parameter_updates

        self.train = theano.function(
                     inputs=[data,
                             l_rate,
                             mu,
                             d_prob],
                     outputs=cost,
                     updates=update(params, grads)
        )

        self.get_hidden = theano.function(
                          inputs=[data,
                                  d_prob],
                          outputs=hid
        )
        #theano.printing.debugprint(self.get_hidden, print_type=True,
        #    file='./da_get_hidden_fn.txt')
