import theano
import theano.tensor as T
import numpy as np
import random
import func

class RNN_layers: 
    def __init__(self , W , Wh , b , n_neurals ,in_seq , batch_size , act_type):
        
        def step(x_t , out_tm1 ): 
            out_t = func.activation_func( x_t  
                                        + T.dot( out_tm1 , Wh) + b.dimshuffle('x',0) , act_type)
            #out_t = ( T.dot( x_t , W ) + T.dot (out_tm1 , Wh) + b )
            return out_t  
        
        z_seq = T.dot( in_seq , W )
        a_init = theano.shared(np.asarray ( np.zeros( ( batch_size , n_neurals ) ) , 'float32' ))
  
        out_seq , _ = theano.scan(
                                fn=step, 
                                sequences =  z_seq , 
                                outputs_info =  [ a_init  ] 
                                )
        
        self.layer_out = out_seq
        
        self.test = theano.function(
                            inputs =  [in_seq] , 
                            outputs = out_seq )

class RNN_first_layer:
    def __init__(self , Wi , Whi , bi , n_neurals , x_seq , batch_size , act_type):
        def step(z_t , out_tm1):
            out_t = func.activation_func( z_t + T.dot(out_tm1 , Whi) + bi.dimshuffle('x' , 0 ) , act_type )
            return out_t

        z_seq = T.dot( x_seq , Wi )
        a_init = theano.shared(np.asarray ( np.zeros( ( batch_size , n_neurals ) ) , 'float32' ))

        out_seq , _ = theano.scan( fn=step,
                                  sequences = z_seq , 
                                  outputs_info = [ a_init ],
                                  truncate_gradient= -1 )
        self.layer_out = out_seq
   
     
class RNN_last_layer:
    def __init__(self , Wo , bo , a_seq ):

        y_seq = T.dot(a_seq , Wo) + bo.dimshuffle('x',0)
        self.layer_out = y_seq 
