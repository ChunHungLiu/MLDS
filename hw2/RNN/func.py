import theano
import theano.tensor as T
import numpy as np

def activation_func(z , act_type):
    if act_type == "ReLU":
        a = T.switch(z<0 , 0 , z )
    elif act_type == "Sigmoid":
        z = T.switch(z<-70, -70, z)
        a = 1/(1+T.exp(-z))
    elif act_type == "tanh":
        a = T.tanh(z)
    else:
        a = z
    return T.cast(a , 'float32')


def cost_func(y , y_hat , cost_type):
    #a = ( T.exp(y) / T.sum(T.exp(y) ))
    #cost = T.sum(( y - y_hat)**2) #/ T.cast ( y_hat.shape[1], 'float32')
    if cost_type == "EU":
        #a.append( 1/(1 + T.exp(-z[idx])) )
        cost = T.sum(( y - y_hat)**2) / T.cast(y_hat.shape[1], 'float32')
    elif cost_type == "CE":
        #y_sum = T.sum( T.exp(y) , axis = 2 )
        #y_reshape = y_sum.reshape( (y_sum.shape[0] , y_sum.shape[1] , 1) )
        #a = ( T.exp(y) / T.repeat  (y_reshape , y_hat.shape[2] , axis = 2  ))
        #a = (T.exp(y) / T.sum(T.exp(y)))
        #a = softmax(y)
        multi = T.sum( (y*y_hat) , axis = 2 )
        after_sw = T.switch(T.eq(multi,0) , 0.9 , multi)
        cost = - T.sum(T.log( (after_sw ) )  ) / T.cast(y_hat.shape[1],'float32')
    return cost

def softmax( y ):
    y_max = T.max(y , axis = 2)
    y_max_rep = y_max.reshape( ( y_max.shape[0] , y_max.shape[1] , 1))
    y_opt = y - T.repeat (y_max_rep , y.shape[2] , axis = 2)
    y_sum = T.sum( T.exp(y_opt) , axis = 2 )
    y_reshape = y_sum.reshape( (y_sum.shape[0] , y_sum.shape[1] , 1) )
    a = ( T.exp(y_opt) / T.repeat  (y_reshape , y.shape[2] , axis = 2  ))
    return a
