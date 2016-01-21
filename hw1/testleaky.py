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
#MODEL = "baseline_build26"
MODEL = "CE_DATA_fbank_LABEL_phoneme48_HIDDEN_LAYERS_1024-1024_L_RATE_0.001_MOMENTUM_0.9_DROPOUT_0.1_EPOCH_500_at_200"
TEST_ROOT = './Data'
TEST = '/fbank/test.ark'

#TEST = './train_ant.ark'

PREDICTION_ROOT ='./result/prediction/'
PREDICTION = MODEL + '.csv'

########################
#  load DNN open file  #
########################

ACT_FUNC="leakyReLU"
COST_FUNC ="CE"
layers,Ws,bs = pickle.load(open(MODEL_ROOT+MODEL,'rb')) 
nn = DNN(layers,Ws,bs,act=ACT_FUNC,cost=COST_FUNC)
#
nn.rescale_params(0.9)
#
MODEL = "DATA_fbank_LABEL_phoneme48_HIDDEN_LAYERS_1024-1024-1024-1024_L_RATE_0.001_MOMENTUM_0.9_DROPOUT_0_EPOCH_100"
TEST_DATA,VAL_DATA = readfile( TEST_ROOT+TEST,1 )
PRED_FILE = open( PREDICTION_ROOT + PREDICTION ,'wb')

# Get Dictionaries
Phone48 = load_list39to48()
PhoneMap48to39 = load_dict_48to39()

# For CSV
HEADER = ["Id","Prediction"]

########################
#       Predict        #
########################


IDs,Feats = SepIDnFeat(TEST_DATA)

x = np.asarray(Feats,dtype='float32').T
y = nn.test(x)

maxpositions = np.argmax(y,axis=0)
output = [ PhoneMap48to39[Phone48[pos]] for pos in maxpositions ]

c = csv.writer(PRED_FILE,delimiter =',')
c.writerow(HEADER)
c.writerows(zip(IDs,output))

PRED_FILE.close()
