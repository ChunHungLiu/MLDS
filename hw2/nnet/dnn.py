import theano
import theano.tensor as T
import numpy as np
import pickle
import random

from itertools import izip
from theano.tensor.shared_randomstreams import RandomStreams



class DNN:
    def __name__(self):
        return "baseline"

    def __init__(self, layers, Ws=None, bs=None,m_norm=None, act="ReLU", cost="CE",momentum_type="adagrad"):
        # validation.
        if Ws is not None and bs is not None:
            assert len(layers) == len(Ws) and len(layers) == len(bs)
        # layers
        self.layers = layers

        self.rng = RandomStreams()

        # hyperparameters:
        l_rate = T.scalar(dtype='float32')
        momentum  = T.scalar(dtype='float32')
        retain_prob = T.scalar(dtype='float32')
        input_r_prob = T.scalar(dtype='float32')
        maxnorm = np.cast['float32'](m_norm)

	    # layer inputs and outputs
        x = T.matrix(dtype='float32')
        y_hat = T.matrix(dtype='float32')
 
        masks = [ self.rng.binomial(size=x.shape,
                                    n=1, p=input_r_prob,
                                    dtype='float32') ]
        z = [ None   ] # z[0] = None
        a = [ masks[0] * x ] # a[0] = dropped input

        # Weight matrices and bias vectors
        self.W = [ None ]
        self.b = [ None ]

        #init = np.sqrt(6 / self.layers[idx+1])
        init = 0.1
        for idx in range(1,len(self.layers)-1):
            self.W.append( theano.shared(np.asarray(
                np.random.uniform(-init,init,
                    size=(self.layers[idx], self.layers[idx-1])),'float32') ))
            # max-norm. c
            if maxnorm is not None:
                w = self.W[idx].get_value()
                w_sum = (w**2).sum(axis=0)
                w[:, w_sum>maxnorm] = \
                    w[:, w_sum>maxnorm]*np.sqrt(maxnorm) / w_sum[w_sum>maxnorm]
                self.W[idx].set_value(np.asarray(w,'float32'))

            self.b.append( theano.shared(np.asarray(
                np.random.uniform(-init,init,size=(self.layers[idx])),'float32') ))

            if Ws is not None:
                self.W[idx].set_value( Ws[idx].get_value() ) 
            if bs is not None:
                self.b[idx].set_value( bs[idx].get_value() )

            z.append( T.dot(self.W[idx],a[idx-1]) + self.b[idx].dimshuffle(0,'x') )
            original = None
            if act == "ReLU":
                original = T.switch(z[idx]<0, 0, z[idx])
            elif act == "sigmoid":
                original = 1/(1 + T.exp(-z[idx]))
            elif act == "leakyReLU":
                original = T.switch(z[idx]<0, np.cast['float32'](0.01)*z[idx], z[idx])
                
            

            masks.append( self.rng.binomial(size=original.shape,
                                            n=1, p=retain_prob,
                                            dtype='float32') )
            a.append( masks[idx] * original)
        # output layer 
        idx = len(self.layers)-1
        self.W.append( theano.shared(np.asarray(
            np.random.uniform(-init,init,
                size=(self.layers[idx], self.layers[idx-1])),'float32') ))
        # max-norm. c
        w = self.W[idx].get_value()
        w_sum = (w**2).sum(axis=0)
        w[:, w_sum>maxnorm] = \
            w[:, w_sum>maxnorm]*np.sqrt(maxnorm) / w_sum[w_sum>maxnorm]
        self.W[idx].set_value(w)

        self.b.append( theano.shared(np.asarray(
            np.random.uniform(-init,init,size=(self.layers[idx])),'float32') ))
        
        if Ws is not None:
            self.W[idx].set_value( Ws[idx].get_value() ) 
        if bs is not None:
            self.b[idx].set_value( bs[idx].get_value() )

        z.append( T.dot(self.W[idx],a[idx-1]) + self.b[idx].dimshuffle(0,'x') )


        if cost == "EU":
            a.append( 1/(1 + T.exp(-z[idx])) )
            cost = T.sum(( a[-1] - y_hat)**2) / T.cast(x.shape[1], 'float32')
        elif cost == "CE": 
            a.append( T.exp(z[idx]) / T.sum(T.exp(z[idx])) )
            cost = - T.sum(T.log(a[-1]+1e-6)*y_hat)/ T.cast(x.shape[1],'float32')

        assert len(a) == len(self.layers)
        parameters = self.W[1:] + self.b[1:]
        prev_updates = []
        for param in parameters:
            prev_updates.append( theano.shared(
                np.asarray(
                    np.zeros(param.get_value().shape),
                'float32' )
            ))
        sq_sum_grad = []
        if momentum_type == "adagrad" or momentum_type == 'rmsprop':
            for param in parameters:
                sq_sum_grad.append( theano.shared(
                    np.asarray(
                        np.zeros(param.get_value().shape),
                    'float32' )
                ))

        gradients = T.grad(cost, parameters)
        self.debug = prev_updates

        # Define Parameter Updates
        def update(parameters,gradients):
            parameter_updates = None
            if momentum_type == "vanilla":
                parameter_updates = [ (p, p - l_rate*g + momentum*v)  \
                    for p,g,v in izip(parameters,gradients,prev_updates) ]
                parameter_updates += [ (v, -l_rate*g + momentum*v) \
                    for g,v in izip(gradients,prev_updates) ]
                return parameter_updates
            elif momentum_type == "adagrad":
                parameter_updates = [ (p, p - l_rate/np.sqrt(ssg)*g)  
                    if ssg.get_value().sum() != 0 else (p, p-l_rate*g) \
                    for p,g,ssg in izip(parameters,gradients,sq_sum_grad) ]
                parameter_updates += [ (ssg, ssg + g**2)\
                    for g,ssg in izip(gradients,sq_sum_grad)]
                return parameter_updates
            elif momentum_type == "rmsprop":
                parameter_updates = [ (p, p - l_rate/np.sqrt(ssg)*g)  
                    if ssg.get_value().sum() != 0 else (p, p-l_rate*g) \
                    for p,g,ssg in izip(parameters,gradients,sq_sum_grad) ]
                parameter_updates += [ (ssg, 0.9*ssg + 0.1*(g**2))\
                    for g,ssg in izip(gradients,sq_sum_grad)]
                return parameter_updates

        self.train = theano.function(
                        inputs = [ x, y_hat,
                                   l_rate,
                                   momentum,
                                   retain_prob,
                                   input_r_prob ],
                        updates = update(parameters, gradients),
                        outputs = cost,
                        allow_input_downcast = True
        )
        self.test = theano.function(
                       inputs = [ x,
                                  theano.Param(retain_prob, default=1),
                                  theano.Param(input_r_prob, default=1)],
                       outputs = a[-1]
        )
    
    def rescale_params(self, scale):
        for W in self.W[1:]:
            W.set_value( 
                W.get_value() * np.cast['float32'](scale) )
        for b in self.b[1:]:
            b.set_value(
                b.get_value() * np.cast['float32'](scale) )
    '''
    def load_params(self, Ws, bs):
        assert len(Ws) == len(self.W)-1
        assert len(bs) == len(self.b)-1
        for i,W in enumerate(Ws[1:]):
            idx = i+1
            self.W[idx].set_value( W.get_value() )
        for i,b in enumerate(bs[1:]):
            idx = i+1
            self.b[idx].set_value( b.get_value() )
    '''
