import numpy as np

from keras.models import Sequential
from keras.layers.core import * #Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import SimpleRNN , LSTM
from keras.optimizers import SGD , Adagrad


def keras_model( batch_dim , image_vector=1000 , word_vector=300 ):

    LSTM_layers = 1 
    LSTM_units  = 300
    #DNN_units   = [4096 ,2048 , 1024 , 512 , 256,128]
    DNN_units   = [ 2048, 2048 ]
    
    question_LSTM = Sequential()
    layer_Mask_q = Masking(mask_value=-999, input_shape=(batch_dim, 300))
    layer_LSTM_q = LSTM(LSTM_units) # exp other act
    question_LSTM.add(layer_Mask_q)
    question_LSTM.add(layer_LSTM_q)

    opt_LSTM_1 = Sequential()
    layer_Mask_1 = Masking(mask_value=-999, input_shape=(batch_dim, 300))
    layer_LSTM_1 = LSTM(LSTM_units)
    opt_LSTM_1.add(layer_Mask_1)
    opt_LSTM_1.add(layer_LSTM_1)

    opt_LSTM_2 = Sequential()
    layer_Mask_2 = Masking(mask_value=-999, input_shape=(batch_dim, 300))
    layer_LSTM_2 = LSTM(LSTM_units)
    opt_LSTM_2.add(layer_Mask_2)
    opt_LSTM_2.add(layer_LSTM_2)

    opt_LSTM_3 = Sequential()
    layer_Mask_3 = Masking(mask_value=-999, input_shape=(batch_dim, 300))
    layer_LSTM_3 = LSTM(LSTM_units)
    opt_LSTM_3.add(layer_Mask_3)
    opt_LSTM_3.add(layer_LSTM_3)

    opt_LSTM_4 = Sequential()
    layer_Mask_4 = Masking(mask_value=-999, input_shape=(batch_dim, 300))
    layer_LSTM_4 = LSTM(LSTM_units)
    opt_LSTM_4.add(layer_Mask_4)
    opt_LSTM_4.add(layer_LSTM_4)

    opt_LSTM_5 = Sequential()
    layer_Mask_5 = Masking(mask_value=-999, input_shape=(batch_dim, 300))
    layer_LSTM_5 = LSTM(LSTM_units)
    opt_LSTM_5.add(layer_Mask_5)
    opt_LSTM_5.add(layer_LSTM_5)

    image_model = Sequential()
    image_model.add(Reshape(input_shape = (image_vector, ), dims = (image_vector , ) ))

    model = Sequential()
    model.add(Merge([ image_model, question_LSTM, opt_LSTM_1, opt_LSTM_2, 
                      opt_LSTM_3, opt_LSTM_4, opt_LSTM_5], 
                    mode='concat', concat_axis=1))
    layer_pre_DNN = Dense(DNN_units[0] , init = 'uniform')
    layer_pre_DNN_act = Activation('relu')
    layer_pre_DNN_dro = Dropout(p=0.5)
    
    layer_DNN_1 = Dense(DNN_units[1] , init = 'uniform')
    layer_DNN_1_act = Activation('relu')
    layer_DNN_1_dro = Dropout(p=0.5)

    '''
    layer_DNN_2 = Dense(DNN_units[2] , init = 'uniform')
    layer_DNN_2_act = Activation('relu')
    layer_DNN_2_dro = Dropout(p=0.5)

    layer_DNN_3     = Dense(DNN_units[3] , init = 'uniform')
    layer_DNN_3_act = Activation('relu')
    layer_DNN_3_dro = Dropout(p=0.5)
    
    layer_DNN_4     = Dense(DNN_units[4] , init = 'uniform')
    layer_DNN_4_act = Activation('relu')
    layer_DNN_4_dro = Dropout(p=0.5)
    
    layer_DNN_5     = Dense(DNN_units[5] , init = 'uniform')
    layer_DNN_5_act = Activation('relu')
    layer_DNN_5_dro = Dropout(p=0.5)
    '''
    
    layer_out     = Dense(5)
    layer_softmax = Activation('softmax')

    model.add(layer_pre_DNN)
    model.add(layer_pre_DNN_act)
    model.add(layer_pre_DNN_dro)
    model.add(layer_DNN_1)
    model.add(layer_DNN_1_act)
    model.add(layer_DNN_1_dro)
    '''
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
    '''
    model.add(layer_out)
    model.add(layer_softmax)

    model.compile(loss='categorical_crossentropy', optimizer='adagrad')
    
    #model.fit([question , a_1 , a_2 , a_3 , a_4 , a_5 , image] , y_train , nb_epoch = 100 , batch_size = 1  )
    return model
'''
import cPickle as pickle

ej_case = 'EJ_case.pkl'
question_input, ans1_input, ans2_input, ans3_input, ans4_input, ans5_input, image_input, solution = pickle.load(open(ej_case,'rb'))

image_v = np.array([image_input[0][0]])
print image_v.shape
print solution.shape
print solution
my_model = keras_model(100)
my_model.fit( [question_input, ans1_input, ans2_input, ans3_input, ans4_input, ans5_input, image_v] ,solution , nb_epoch = 100 , batch_size = 1  )

print my_model.predict( [question_input, ans1_input, ans2_input, ans3_input, ans4_input, ans5_input, image_v]  )
'''
# demo of noLSTM
'''
X = np.ones((2,2800))
Y = np.array([[0,0,1,0,0],[0,0,1,0,0]])
print X.shape
print Y.shape
my_model = keras_model(100)
for t in range(100):
    print my_model.train_on_batch(X,Y)
print my_model.predict_on_batch(X)
'''
