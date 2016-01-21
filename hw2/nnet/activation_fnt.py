import theano
import theano.tensor as T
import numpy as p

# sigmoid
def sigmoid():
    sig_in = T.scalar()
    sig_out = 1/(1 + T.exp(-sig_in))
    return theano.function( inputs=[sig_in], outputs=sig_out)

# tanh
def tanh():
    return

# ReLU

def ReLU(default = ""):
    z = T.scalar()
    a = T.switch(z<0,0.01*z,z)
    return theano.function( inputs = [z], outputs = a)
