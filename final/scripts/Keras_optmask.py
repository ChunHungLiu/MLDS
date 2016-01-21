import numpy as np
import pdb
from keras.models import Sequential
from keras.layers.core import * #Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import *#SimpleRNN , LSTM
from keras.optimizers import SGD , Adagrad

def keras_model():
    word_vector = 300
    LSTM_units  = 300
    
    model_filter = Sequential()
    model_data   = Sequential()
    
    model = Sequential() 
    
    ####structure of the filter's network
    ####the last layer's neural unit must be equal to the data dimension to be masked
    model_filter.add( Dense( 4096 , input_dim = 2800 ) )
    model_filter.add( Activation('relu') )
    model_filter.add( Dropout(p=0.5) )
    
    model_filter.add( Dense( 2048) )
    model_filter.add( Activation('relu') )
    model_filter.add( Dropout(p=0.5) )
    
    model_filter.add( Dense( 2048) )
    model_filter.add( Activation('relu') )
    model_filter.add( Dropout(p=0.5) )

    model_filter.add( Dense( 2048) )
    model_filter.add( Activation('relu') )
    model_filter.add( Dropout(p=0.5) )
    
    model_filter.add( Dense( 1024) )
    model_filter.add( Activation('relu') )
    model_filter.add( Dropout(p=0.5) )
   
    model_filter.add( Dense( 1024) )
    model_filter.add( Activation('relu') )
    model_filter.add( Dropout(p=0.5) )
   
    model_filter.add( Dense( 1024) )
    model_filter.add( Activation('relu') )
    model_filter.add( Dropout(p=0.5) )
   
    model_filter.add( Dense( 1000 ,input_dim = 2800) )
    # TO exp
    model_filter.add( Activation('softmax') )
    
    ####end of the filter's network
    
    # Set input data model as a buffer 
    layer_data = Reshape( input_shape = ( 2800 , ) , dims = (2800 , ) )
    model_data.add(layer_data)
    
    #### overall model 
    model.add(Merge( [model_filter , model_data] ,  mode='concat', concat_axis=1 ) )
    #### This layer means .* of model_filter's output and model_data[:data_dim]
    #### So the dimension of above two should be equal
    #### In this case, they are 1000
    model.add(Matched_filter_opt( data_dim = 1000))
    
    #### the DNN
    #model.add(Dense(50))
    model.add( Dense(2048) )
    model.add( Activation('relu') )
    model.add( Dropout(p=0.5) )
    
    model.add( Dense(2048) )
    model.add( Activation('relu') )
    model.add( Dropout(p=0.5) )
    
    model.add( Dense(1024) )
    model.add( Activation('relu') )
    model.add( Dropout(p=0.5) )
    
    model.add( Dense(1024) )
    model.add( Activation('relu') )
    model.add( Dropout(p=0.5) )
    
    model.add( Dense(1024) )
    model.add( Activation('relu') )
    model.add( Dropout(p=0.5) )
    
    model.add( Dense(512) )
    model.add( Activation('relu') )
    model.add( Dropout(p=0.5) )

    model.add( Dense(512) )
    model.add( Activation('relu') )
    model.add( Dropout(p=0.5) )
    
    model.add( Dense(256) )
    model.add( Activation('relu') )
    model.add( Dropout(p=0.5) )
    # --- output ---
    model.add( Dense(5) )
    model.add( Activation('softmax') )
    
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')
    return model
'''
Starting testing
'''
'''
import cPickle as pickle

print "Starting ..." 
question_input, ans1_input, ans2_input, ans3_input, ans4_input, ans5_input, image_input, solution = pickle.load(open(ej_case,'rb'))

image_v    = np.array([ image_input[0][0] ])
print solution.shape

temp = np.hstack(( question_input[0][0], ans1_input[0][0] , ans2_input[0][0] ,ans3_input[0][0] ,ans4_input[0][0] ,ans5_input[0][0] ))
temp = np.array([[temp]])
print temp[0].shape
print image_v.shape
input_data = np.hstack(( image_v , temp[0] ))
print input_data.shape


for n in range(100):
    #print "model_1", model.predict_on_batch( [ ans1_input , image_input] , solution )
    print model.train_on_batch( [ input_data , image_v] , solution )
    #print model_1.train_on_batch(ans1_input , solution)

print model.predict_on_batch( [ input_data , image_v ])
#print model_1.predict_on_batch(ans1_input)
'''
