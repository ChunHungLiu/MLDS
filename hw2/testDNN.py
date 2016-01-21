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
MODEL = "DATA_fbank_LABEL_phonemeState_HIDDEN_LAYERS_2048-2048-2048_L_RATE_0.01_MOMENTUM_0.9_DROPOUT_0.1_EPOCH_200_at_150"

TEST_ROOT = './Data/fbank/'
TEST = 'test.ark'

#TEST = './train_ant.ark'

PREDICTION_ROOT ='./result/prediction/'
PREDICTION = MODEL + '.csv'

########################
#  load DNN open file  #
########################

layers,Ws,bs = pickle.load(open(MODEL_ROOT+MODEL,'rb'))
nn = DNN(layers,Ws,bs)

TEST_DATA,VAL_DATA = readfile_( TEST_ROOT+TEST,1 )
PRED_FILE = open( PREDICTION_ROOT + PREDICTION ,'wb')

# Get Dictionaries
Phone48 = load_liststateto48()
PhoneMap48to39 = load_dict_48to39()

# For CSV
HEADER = ["Id","Prediction"]

########################
#       Predict        #
########################


IDs,Feats = SepIDnFeat(TEST_DATA)
del TEST_DATA

x = np.asarray(Feats,dtype='float32')
x = np.transpose(x)

y = nn.test(x)

maxpositions = np.argmax(y,axis=0)
output = [ PhoneMap48to39[Phone48[pos]] for pos in maxpositions ]

del x
c = csv.writer(PRED_FILE,delimiter =',')
c.writerow(HEADER)
c.writerows(zip(IDs,output))

PRED_FILE.close()
