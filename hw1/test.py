import csv
import pickle
import pdb

from nnet import *
#from nnet.dA import dA
from nnet.dnn import DNN
from reader import *

########################
#   Define Paramters   #
########################

MODEL_ROOT = "./result/model/"
MODEL = "DATA_fbank_LABEL_phonemeState_HIDDEN_LAYERS_2048-2048-2048_L_RATE_0.01_MOMENTUM_0.9_DROPOUT_0.1_EPOCH_360_at_240"

TEST_ROOT = './Data/fbank/'
#TEST = '/fbank/test.ark'

TEST = 'test.ark'

PREDICTION_ROOT ='./result/prediction/'
PREDICTION = MODEL + '.csv'

########################
#  load DNN open file  #
########################

layers,Ws,bs = pickle.load(open(MODEL_ROOT+MODEL,'rb')) 
nn = DNN(layers,Ws,bs)

IDs,TEST_DATA,VAL_DATA = readfile_for_test( TEST_ROOT+TEST,1 )
PRED_FILE = open( PREDICTION_ROOT + PREDICTION ,'wb')

# Get Dictionaries
PhoneState = load_liststateto48()
>>>>>>> c8f5d46a94a4e5c1bd918860a07a7585738c306d
PhoneMap48to39 = load_dict_48to39()

# For CSV
HEADER = ["Id","Prediction"]

########################
#       Predict        #
########################

x = np.asarray(TEST_DATA,dtype='float32').T
>>>>>>> c8f5d46a94a4e5c1bd918860a07a7585738c306d
y = nn.test(x)

maxpositions = np.argmax(y,axis=0)
output = [ PhoneMap48to39[PhoneState[pos]] for pos in maxpositions ]

del x
c = csv.writer(PRED_FILE,delimiter =',')
c.writerow(HEADER)
c.writerows(zip(IDs,output))

PRED_FILE.close()
