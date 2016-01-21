import numpy as np

from keras.models import Sequential
from keras.layers.core import * #Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import *#SimpleRNN , LSTM
from keras.optimizers import SGD , Adagrad


word_vector = 300
LSTM_units  = 300

model_1 = Sequential()
model_2 = Sequential()

layer_a1 = LSTM ( 5 , input_shape = (30 , word_vector) , return_sequences= False )
layer_a2 = beShared_LSTM(layer_a1 , input_shape = (30 , word_vector))

model_1.add(layer_a1)
model_2.add(layer_a2)



model_1.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model_2.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
Starting testing
'''

import cPickle as pickle

print "Starting ..." 

ej_case = 'EJ_case.pkl'
question_input, ans1_input, ans2_input, ans3_input, ans4_input, ans5_input, image_input, solution = pickle.load(open(ej_case,'rb'))

print ans1_input.shape
print solution.shape

for n in range(10):
    print "model_2", model_2.predict_on_batch(ans1_input)
    print "model_1", model_1.predict_on_batch(ans1_input)
    print model_1.train_on_batch(ans1_input , solution)

print model_2.predict_on_batch(ans1_input)
print model_1.predict_on_batch(ans1_input)
