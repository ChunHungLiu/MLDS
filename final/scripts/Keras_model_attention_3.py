import numpy as np

from keras.models import Sequential
from keras.layers.core import * #Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import *#SimpleRNN , LSTM
from keras.optimizers import SGD , Adagrad


def keras_model(  image_vector=1000 , word_vector=300 ):

    DNN_units   = [2048 , 2048 , 1024 , 1024 , 512, 512 ]

    model = Sequential()
    
    attention_base = Matched_filter_3(input_dim = 2800)
   
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
    
    layer_DNN_6     = Dense(DNN_units[5] , init = 'uniform')
    layer_DNN_6_act = Activation('relu')
    layer_DNN_6_dro = Dropout(p=0.5)

    layer_out     = Dense(5)
    layer_softmax = Activation('softmax')

    model.add(attention_base)
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
    model.add(layer_DNN_6)
    model.add(layer_DNN_6_act)
    model.add(layer_DNN_6_dro)
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
print ans1_input.shape
print ans1_input[0][0].shape
print solution.shape
print solution


temp = np.hstack(( question_input[0][0], ans1_input[0][0] , ans2_input[0][0] ,ans3_input[0][0] ,ans4_input[0][0] ,ans5_input[0][0] ))
temp = np.array([[temp]])
print temp[0].shape
print image_v.shape
input_data = np.hstack(( image_v , temp[0] ))
print input_data.shape

my_model = keras_model(30)

for t in range(100):
    print my_model.train_on_batch(input_data ,solution  )
#    my_model.save_weights('test'+str(t)+'.hdf5')
#my_model.fit( [question_input, ans1_input, ans2_input, ans3_input, ans4_input, ans5_input, image_v] ,solution , nb_epoch = 100 , batch_size = 1  )
#my_model.fit( [question_input, ans5_input, image_v] ,solution , nb_epoch = 100 , batch_size = 1  )

print my_model.predict( [image_v , question_input, ans1_input, ans2_input, ans3_input, ans4_input, ans5_input]  )
'''
