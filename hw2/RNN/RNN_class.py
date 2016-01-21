import theano
import theano.tensor as T
import numpy as np
import random
from itertools import izip

from RNN_layers import RNN_layers
import func as F
        
class RNN_net:
    def __init__(self , layers , Ws = None , Whs = None , bs = None , batch_size = 1 ,
                    momentum_type = "None" , act_type = "ReLU" , cost_type = "CE"  ):
       #test parameter define ( should be inputs later.)
        self.layers        = layers 
        self.batch_size    = batch_size 
        l_rate             = T.scalar(dtype='float32') # np.float32(0.0001) 
        init               = np.float32(0.1)     
        rms_alpha          = T.scalar(dtype='float32') # np.float32(0.9)
        clip_range         = T.scalar(dtype='float32') 
       # validation.  
        if Ws is not None and bs is not None and Whs is not None:
            assert len(layers) == len(Ws) and len(layers) == len(bs) and len(layers) == len(Whs)
       # train input 
        x_seq = T.tensor3(dtype='float32')
        y_hat = T.tensor3(dtype='float32')
        mask  = T.tensor3(dtype='float32')      
        #mask  = T.TensorType( 'float32' , (False,False,True) )       

       # train parameter initialization
        #self.Wi = 
        #self.Wo = 
        #self.Bo =
        self.W =  [ None ]
        self.Wh = [ None ]
        self.b =  [ None ]

        a_seq = [ x_seq ] 
        ls = [ None ]

        for idx in range( len(self.layers)-1 ):
            # init b , Wh , W 
            #self.b.append ( theano.shared(np.asarray (np.random.uniform(-init , init , size = ( self.layers[idx+1] )) , 'float32')))   
            self.b.append ( theano.shared(np.asarray (np.zeros(( self.layers[idx+1] )) , 'float32')))   
            self.Wh.append (theano.shared(np.asarray (np.identity(self.layers[idx+1]), 'float32')) )  
            #if self.layers[idx] == self.layers[idx+1]:
            #    self.W.append(theano.shared(np.asarray ( np.identity(self.layers[idx]) , 'float32' )))            
            #else:
            self.W.append(theano.shared(np.asarray ( np.random.uniform(-init , init , size = ( self.layers[idx] , self.layers[idx+1] )), 'float32'  )  ))
            # import the  model from outside
            if Ws is not None:
                self.W[idx].set_value( Ws[idx].get_value() ) 
            if bs is not None:
                self.b[idx].set_value( bs[idx].get_value() )
            if Whs is not None:
                self.Wh[idx].set_value( Whs[idx].get_value() ) 
            
            # declaration a RNN layer
            temp_layers = RNN_layers(self.W[idx+1] , self.Wh[idx+1] , self.b[idx+1] , self.layers[idx+1] , a_seq[idx] , self.batch_size  , act_type) 
            ls.append(temp_layers)   
            # output the 'a' of RNN layers 
            a_seq.append(temp_layers.layer_out)
       
       # define parameters 
        parameters = self.W[1:] + self.Wh[1:] + self.b[1:] 
    
       # define what are outputs.
        y_seq = a_seq[-1]
        y_out = y_seq * T.addbroadcast( mask , 2  )
       # define cost 
        cost = F.cost_func( y_out , y_hat , cost_type ) 
       # compute gradient    
        gradients = T.grad(cost , parameters )
        gradient = [ ]
        for idx in range(len(gradients)):
            gradient.append(T.clip(gradients[idx] , -clip_range , clip_range) ) 

        # for rmsprop
        sq_sum_grad = []
        for param in parameters:
            sq_sum_grad.append( theano.shared(
                np.asarray(
                    np.zeros(param.get_value().shape) , 'float32' ) 
            ))

        def update(parameters , gradients ):
            if momentum_type == "rmsprop":
                parameter_updates = [ (p, p - l_rate*g)  
                    if ssg.get_value().sum() != 0 else (p, p-l_rate*g) \
                    for p,g,ssg in izip(parameters,gradient,sq_sum_grad) ]
                parameter_updates += [ (ssg, rms_alpha*ssg + (np.cast['float32'](1.0)-rms_alpha)*(g**2)  ) \
                           for g , ssg in izip( gradient , sq_sum_grad) ]
                return parameter_updates
            elif momentum_type == "None":
                parameter_updates = [  (p, p - l_rate*g) \
                    for p , g in izip(parameters , gradient ) ]  
                return parameter_updates



       # define theano.functions             
        self.train = theano.function( inputs = [x_seq , y_hat , mask ,
                                                l_rate ,
                                                rms_alpha , 
                                                clip_range 
                                                ] ,
                                        updates = update(parameters , gradient) ,
                                        outputs = cost  )
        
        self.test  = theano.function( inputs = [x_seq , mask ]  , outputs = y_out )
        self.gradi = theano.function( inputs = [x_seq , y_hat ,mask]  , outputs = gradients[0] )
