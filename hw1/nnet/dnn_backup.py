# built-in packages
import pdb
from itertools import izip
try:
   import cPickle as pickle
except:
   import pickle

# 3rd-party packages
import numpy as np
import theano
import theano.tensor as T

class DNN:
    def __init__(self, data_layer, hidden_layers, label_layer, act_fnt, learn):
        '''
            self.layers:   hidden layers

            self.x:               data
            self.Y:               label

            self.z:               neuron input
            self.a:               neuron output

            self.W:               weight matrices
            self.b:               bias terms

            self.layer_prop:      function: weight*input + bias
            self.target_error:    difference between error and label

            self.delta:           delta for backpropagate iteration

            self.W_grad:          weight gradients
            self.b_grad:          bias term gradients

            self.learning_rate:   learning rate
            self.set_activation_func:
                                  set_activation funct
            self._update:         update paramters
        '''
        # Initialize Data
        self.x  = None
        self.Y  = None

        # Neural Network
        # Baseline: [ 69,128,48 ]
        self.layers = data_layer + hidden_layers + label_layer

        self.W  = [ None ]
        self.b  = [ None ]
        self.z  = [ None ]
        self.a  = [ None ]

        self.layer_prop = [ None ]

        #Back Propagation
        self.target_error = None

        self.delta = []
        self.W_grad = [ None ]
        self.b_grad = [ None ]

        for idx, dim in enumerate( zip(self.layers[1:],self.layers) ):
            # Create Values & Model Parameters
            # Initialize weight matrices
            init = 0.001 * np.sqrt(dim[0])
            #self.W.append( theano.shared(np.random.uniform(-init,init,size=dim)))
            self.W.append( theano.shared(np.ones(dim)) )
            self.b.append( theano.shared(np.ones(dim[0])) )

            self.z.append( None )
            self.a.append( None )

            self.W_grad.append( theano.shared(np.zeros(dim)) )
            self.b_grad.append( theano.shared(np.zeros(dim[0])) )

            T_input = T.vector()
            if idx == 0:
                T_z_vec = self.b[idx+1] + T.dot(self.W[idx+1],T_input)
                self.layer_prop.append( theano.function([T_input],T_z_vec))
            else:
                T_z_vec = self.b[idx+1] + T.dot(self.W[idx+1],T_input)
                self.layer_prop.append( theano.function([T_input],T_z_vec))

        # learning rate
        self.learning_rate = learn

        # Perform Operations
        self.set_activation_func()
        # Activation Function
        '''
        self.act_func = np.vectorize(act_fnt)
        T_in = T.scalar()
        T_out = act_fnt(T_in)
        T_out_grad = grad(T_out,T_in)
        self.grad_act_func = np.vectorize(theano.function([T_in],T_out_grad))
        '''
        # self._update = theano.function(inputs = [],updates = self.param_update() )



    def set_activation_func(self):
        z  = T.scalar()
        # Sigmoid
        #a = 1/(1 + T.exp(-z))
        # ReLU
        a = T.switch(z<0,0.01*z,z)

        grad_a = T.grad(a, z)

        self.act_func      = np.vectorize( theano.function( [z], a ) )
        self.grad_act_func = np.vectorize( theano.function( [z], grad_a ) )

    def forward(self, data, label = None):

        self.Y = label
        self.x = self.z[0] = layer_input = data

        for idx in range(1,len(self.W)):
            self.z[idx] = self.layer_prop[idx](layer_input)
            self.a[idx] = layer_input = self.act_func(self.z[idx])
        return self.a[-1]

    def calculate_error(self):
        # Relates to cost function
        # Derivative of Euclidean Distance
        self.target_error = [ float(label - output)/len(self.Y) for label, output in izip(self.Y, self.a[-1]) ]
        # Cross Entropy Realization

    def backpropagate(self):
        # for l in reversed(range(2,len(self.layers))):
        # reversed index of layer [ 4, 5, 3 ] => [ 2, 1 ]
        for l in reversed(range(1,len(self.W))):
            print "l",l
            grad_z = np.array(self.grad_act_func(self.z[l]))
            if l == len(self.W)-1:
                delta = np.array(np.multiply(grad_z,self.target_error))
            else:
                delta = grad_z.reshape(1,len(grad_z)) * \
                        self.W[l+1].T * \
                        delta

            self.W_grad[l] = delta.reshape(len(delta),1) * np.array(self.a[l-1]).reshape(1,len(self.a[l-1]))
            self.b_grad[l] = delta

    def param_update(self):
        w_update_list = [ (w_m, w_m - self.learning_rate * dw_m) \
            for w_m,dw_m in izip(self.W, self.W_grad) ]
        b_update_list = [ (b_m, b_m - self.learning_rate * db_m) \
            for b_m,db_m in izip(self.b, self.b_grad) ]
        return w_update_list + b_update_list

    def update(self):
        self._update()

    def predict(self):
        pass

    def save_model(self, filename):
        with open(filename, 'wb') as out_f:
            try:
                pickle.dump(self, out_f)
            except (EnvironmentError, pickle.PicklingError) as err:
                print "Error: Modal not saved."
            finally:
                print "Modal saved."

    @classmethod
    def load_model(self, filename):
        with open(filename, 'rb') as in_f:
            return pickle.load(in_f)
