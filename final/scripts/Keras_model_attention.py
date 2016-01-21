import numpy as np

from keras.models import Sequential
from keras.layers.core import * #Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import *#SimpleRNN , LSTM
from keras.optimizers import SGD , Adagrad


def keras_model( max_seq_length , image_vector=1000 , word_vector=300 ):

    #LSTM_layers = 1 
    #LSTM_units  = 300
    #DNN_layers  = 3
    DNN_units   = [2048 , 1024 , 512 , 256 , 128 ]
    
    model = Sequential()
    # attention base layers
    
    
    attention_base = Matched_filter(input_dim = 2800)

    layer_DNN_1 = Dense(DNN_units[0] , init = 'uniform')
    layer_DNN_1_act = Activation('relu')
    layer_DNN_1_dro = Dropout(p=0.5)
     
    layer_DNN_2 = Dense(DNN_units[1] , init = 'uniform')
    layer_DNN_2_act = Activation('relu')
    layer_DNN_2_dro = Dropout(p=0.5)

    layer_DNN_3     = Dense(DNN_units[2] , init = 'uniform')
    layer_DNN_3_act = Activation('relu')
    layer_DNN_3_dro = Dropout(p=0.5)
    
    layer_DNN_4     = Dense(DNN_units[3] , init = 'uniform')
    layer_DNN_4_act = Activation('relu')
    layer_DNN_4_dro = Dropout(p=0.5)
   
    layer_DNN_5     = Dense(DNN_units[4] , init = 'uniform')
    layer_DNN_5_act = Activation('relu')
    layer_DNN_5_dro = Dropout(p=0.5)

    layer_out     = Dense(5)
    layer_softmax = Activation('softmax')

    model.add( attention_base)
    model.add(layer_DNN_1)
    model.add(layer_DNN_1_act)
    model.add(layer_DNN_1_dro)
    model.add(layer_DNN_2)
    model.add(layer_DNN_2_act)
    model.add(layer_DNN_2_dro)
    model.add(layer_DNN_3)
    model.add(layer_DNN_3_act)
    model.add(layer_DNN_3_dro)
    model.add(layer_DNN_4)
    model.add(layer_DNN_4_act)
    model.add(layer_DNN_4_dro)
    model.add(layer_DNN_5)
    model.add(layer_DNN_5_act)
    model.add(layer_DNN_5_dro)
    model.add(layer_out)
    model.add(layer_softmax)
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')
    
    return model
'''
# Below is a demo of using this model.
import cPickle as pickle

print "Starting ..." 

ej_case = 'EJ_case.pkl'
question_input, ans1_input, ans2_input, ans3_input, ans4_input, ans5_input, image_input, solution = pickle.load(open(ej_case,'rb'))

image_v = np.array([image_input[0][0]])
print image_v.shape
print solution.shape
print solution
my_model = keras_model(30)
for t in range(100):
    print my_model.train_on_batch( [ image_v, question_input, ans1_input, ans2_input, ans3_input, ans4_input, ans5_input] ,solution  )
#    my_model.save_weights('test'+str(t)+'.hdf5')
#my_model.fit( [question_input, ans1_input, ans2_input, ans3_input, ans4_input, ans5_input, image_v] ,solution , nb_epoch = 100 , batch_size = 1  )
#my_model.fit( [question_input, ans5_input, image_v] ,solution , nb_epoch = 100 , batch_size = 1  )

print my_model.predict( [image_v , question_input, ans1_input, ans2_input, ans3_input, ans4_input, ans5_input]  )
'''

