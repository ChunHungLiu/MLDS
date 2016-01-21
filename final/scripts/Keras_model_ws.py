import numpy as np

from keras.models import Sequential
from keras.layers.core import * #Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import *#SimpleRNN , LSTM
from keras.optimizers import SGD , Adagrad


def keras_model( max_seq_length , image_vector=1000 , word_vector=300 ):

    LSTM_layers = 1 
    LSTM_units  = 300
    DNN_layers  = 3
    DNN_units   = 32

    # question 
    question_model = Sequential()
    layer_q1 = LSTM ( LSTM_units , input_shape = (max_seq_length , word_vector) , return_sequences= False )
    question_model.add(layer_q1)
    
    # answer-1
    answer_1_model = Sequential()
    layer_a1 = LSTM ( LSTM_units , input_shape = (max_seq_length , word_vector) , return_sequences= False )
    answer_1_model.add(layer_a1)
    # answer-2
    answer_2_model = Sequential()
    layer_a2 = beShared_LSTM(layer_a1 , input_shape = (max_seq_length , word_vector))
    answer_2_model.add(layer_a2)
    # answer-3
    answer_3_model = Sequential()
    layer_a3 = beShared_LSTM(layer_a1 , input_shape = (max_seq_length , word_vector))
    answer_3_model.add(layer_a3)
    # answer-4
    answer_4_model = Sequential()
    layer_a4 = beShared_LSTM(layer_a1 , input_shape = (max_seq_length , word_vector))
    answer_4_model.add(layer_a4)
    # answer-5
    answer_5_model = Sequential()
    layer_a5 = beShared_LSTM(layer_a1 , input_shape = (max_seq_length , word_vector))
    answer_5_model.add(layer_a5)

    #image
    image_model = Sequential()
    image_model.add(Reshape(input_shape = (image_vector , ) , dims = (image_vector , ) ))
    #image_model.add(Extent_mask(image_vector))

    model = Sequential()
    model.add(Merge([question_model , answer_1_model , 
                                      answer_2_model ,
                                      answer_3_model , 
                                      answer_4_model , 
                                      answer_5_model , image_model], mode='concat', concat_axis=1))
    layer_DNN_1 = Dense(DNN_units , init = 'uniform')
    layer_DNN_1_act = Activation('relu')
    layer_DNN_1_dro = Dropout(p=0.5)
    
    layer_DNN_2 = Dense(DNN_units , init = 'uniform')
    layer_DNN_2_act = Activation('relu')
    layer_DNN_2_dro = Dropout(p=0.5)

    layer_DNN_3     = Dense(DNN_units , init = 'uniform')
    layer_DNN_3_act = Activation('relu')
    layer_DNN_3_dro = Dropout(p=0.5)
    
    layer_out     = Dense(5)
    layer_softmax = Activation('softmax')

    #attention_base = Matched_filter()
    #model.add( attention_base)

    model.add(layer_DNN_1)
    model.add(layer_DNN_1_act)
    model.add(layer_DNN_1_dro)
    
    model.add(layer_DNN_2)
    model.add(layer_DNN_2_act)
    model.add(layer_DNN_2_dro)
    model.add(layer_DNN_3)
    model.add(layer_DNN_3_act)
    model.add(layer_DNN_3_dro)
    
    model.add(layer_out)
    model.add(layer_softmax)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    return model

# Below is a demo of using this model.
''''
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
    print my_model.train_on_batch( [question_input, ans1_input, ans2_input, ans3_input, ans4_input, ans5_input, image_v] ,solution  )
    my_model.save_weights('test'+str(t)+'.hdf5')
#my_model.fit( [question_input, ans1_input, ans2_input, ans3_input, ans4_input, ans5_input, image_v] ,solution , nb_epoch = 100 , batch_size = 1  )
#my_model.fit( [question_input, ans5_input, image_v] ,solution , nb_epoch = 100 , batch_size = 1  )

print my_model.predict( [question_input, ans1_input, ans2_input, ans3_input, ans4_input, ans5_input, image_v]  )
'''
