from itertools import izip

import numpy as np
import theano
import theano.tensor as T

import func as F
from lstm_layer import *


class LSTM_net:
    def __init__( self, layers, Ws = None, Wis = None, Wfs = None, Wos = None, bs = None, bis = None, bfs = None, bos = None, \
                batch_size = 1, momentum_type = "None", act_type = "ReLU" , cost_type = "EU" ):

        self.layers       = layers
        self.batch_size   = batch_size

        l_rate            = T.scalar('float32')
        init              = np.float32(0.1)
        rms_alpha         = T.scalar('float32') # np.float32(0.9)
        clip_range        = T.scalar('float32')
        momentum          = T.scalar('float32')


        x_seq   = T.tensor3(dtype='float32')
        y_h_seq = T.tensor3(dtype='float32')
        mask    = T.tensor3(dtype='float32')

        self.W  = [ None ]
        self.Wi = [ None ]
        self.Wf = [ None ]
        self.Wo = [ None ]
        self.b  = [ None ]
        self.bi = [ None ]
        self.bf = [ None ]
        self.bo = [ None ]

        a_seq       = [ x_seq ]
        lstm_layers = [ None ]

        parameters = [ self.W, self.Wi, self.Wf, self.Wo,
                       self.b, self.bi, self.bf, self.bo ]

        for idx in xrange(1,len(layers)):
            # Initializing model parameters.
            for i,p in enumerate(parameters):
                if i < 4: # Weight Matrices
                    if idx == len(layers) - 1:
                        p.append( theano.shared( np.random.uniform( -init, init, \
                                size=(layers[idx-1],layers[idx]) ).astype("float32") ))
                    else:
                        p.append( theano.shared( np.random.uniform( -init, init, \
                                size=(layers[idx-1]+2*layers[idx],layers[idx]) ).astype("float32") ))

                else: # bias vectors
                    p.append( theano.shared( np.random.uniform( -init, init, \
                                size = (layers[idx]) ).astype('float32') ))

            # Create LSTM layers and pass in the corresponding parameters.
            if Ws and Wis and Wfs and Wos and bs and bis and bfs and bos:
                layer_params = ( Ws[idx],Wis[idx],Wfs[idx],Wos[idx],bs[idx],bis[idx],bfs[idx],bos[idx] )
            else:
                if idx == len(layers) - 1:
                    layer_params = [ parameters[0][idx] ] + [ None ] * 3 + [ parameters[4][idx] ] + [ None ] * 3
                else:
                    layer_params = [ p[idx] for p in parameters ]

            if idx == len(layers) - 1:
                lstm = LSTM_last_layer( layer_params[0], layer_params[4], a_seq[idx-1], act_type )
            else:
                lstm = LSTMLayer( batch_size, layers[idx-1], layers[idx], a_seq[idx-1], layer_params, act_type )

            a_seq.append( lstm.y_seq )
            lstm_layers.append( lstm )

        y_seq = a_seq[-1]
        y_out = y_seq * T.addbroadcast( mask , 2  )

        if( cost_type == "CE" ):
            y_out = F.softmax(y_out)

        cost = F.cost_func( y_out , y_h_seq , cost_type )

        if Ws and Wis and Wfs and Wos and bs and bis and bfs and bos:
        	parameters = Ws[1:] + Wis[1:-1] + Wfs[1:-1] + Wos[1:-1] + \
				bs[1:] + bis[1:-1] + bfs[1:-1] + bos[1:-1]
        else:
            parameters = self.W[1:] + self.Wi[1:-1] + self.Wf[1:-1] + self.Wo[1:-1]+ \
                     self.b[1:] + self.bi[1:-1] + self.bf[1:-1] + self.bo[1:-1]

        gradients = T.grad(cost , parameters )

        gradient = [ ]
        for idx in range(len(gradients)):
            gradient.append(T.clip(gradients[idx] , -clip_range , clip_range) )

        pre_parameters = []
        for param in parameters:
            pre_parameters.append( theano.shared(
                np.asarray(
                    np.zeros(param.get_value().shape) , 'float32' )
            ))
        # for rmsprop
        sq_sum_grad = []
        for param in parameters:
            sq_sum_grad.append( theano.shared(
                np.asarray(
                    np.zeros(param.get_value().shape) , 'float32' )
            ))
        # for NAG
        pre_update = []
        for param in parameters:
            pre_update.append( theano.shared(
                np.asarray(
                    np.zeros( param.get_value().shape ) , 'float32' )
            ))

        def update(parameters , gradients ):
            if momentum_type == "rmsprop":
                parameter_updates = [ (p, p - l_rate * g / T.sqrt(ssg) )
                    if ssg.get_value().sum() != 0 else (p, p-l_rate*g) \
                    for p,g,ssg in izip(parameters,gradient,sq_sum_grad) ]
                parameter_updates += [ (ssg, rms_alpha*ssg + (np.cast['float32'](1.0)-rms_alpha)*(g**2)  ) \
                           for g , ssg in izip( gradient , sq_sum_grad) ]
                return parameter_updates
            elif momentum_type == "NAG":
                parameter_updates = [ ( pre_p , pre_p + momentum*v - l_rate*g )\
                    for pre_p , g , v in izip(pre_parameters, gradient, pre_update) ]
                parameter_updates += [ ( p , pre_p + 2*( momentum*v - l_rate*g ) ) \
                    for p , pre_p , g , v in izip(parameters, pre_parameters, gradient, pre_update) ]
                parameter_updates += [ ( v , -l_rate*g + momentum*v )\
                    for g , v in izip(gradient , pre_update) ]
                return parameter_updates
            elif momentum_type == "rms+NAG":
                parameter_updates =  [ ( pre_p , pre_p + momentum*v - l_rate*g/T.sqrt(ssg) ) \
                    if ssg.get_value().sum() != 0 else (pre_p , pre_p - l_rate*g + momentum*v ) \
                    for pre_p , g , v , ssg in izip(pre_parameters, gradient, pre_update,sq_sum_grad) ]
                parameter_updates += [ ( p , pre_p + 2*( momentum*v - l_rate*g/T.sqrt(ssg) ) ) \
                    if ssg.get_value().sum() != 0 else ( p , pre_p + 2*( -l_rate*g + momentum*v) ) \
                    for p , pre_p , g , v ,ssg in izip(parameters, pre_parameters, gradient, pre_update , sq_sum_grad) ]
                parameter_updates += [ ( v , -l_rate*g/T.sqrt(ssg) + momentum*v )\
                    if ssg.get_value().sum() != 0 else ( v  , - l_rate*g + momentum*v ) \
                    for g , v , ssg in izip(gradient , pre_update , sq_sum_grad) ]
                parameter_updates += [ (ssg, rms_alpha*ssg + (np.cast['float32'](1.0)-rms_alpha)*(g**2)  ) \
                    for g , ssg in izip( gradient , sq_sum_grad) ]
                return parameter_updates
            elif momentum_type == "None":
                parameter_updates = [ ( p, p - l_rate*g) \
                    for p , g in izip(parameters , gradient ) ]
                return parameter_updates

        self.train = theano.function(
                        inputs  = [ x_seq, y_h_seq, mask, l_rate, rms_alpha ,clip_range, momentum ],
                        outputs = cost,
                        updates = update( parameters, gradients),
                        allow_input_downcast=True
        )

        self.test  = theano.function(
                        inputs = [ x_seq, mask ],
                        outputs = y_out,
                        allow_input_downcast=True
        )
