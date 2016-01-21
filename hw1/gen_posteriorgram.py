import pdb
import pickle

import numpy as np

from nnet import *
#from nnet.dA import dA
from nnet.dnn import DNN
from reader import *

########################
#   Define Paramters   #
########################

MODEL_ROOT = "./result/model/"
MODEL = "DATA_fbank_LABEL_phonemeState_HIDDEN_LAYERS_2048-2048-2048_L_RATE_0.01_MOMENTUM_0.9_DROPOUT_0.1_EPOCH_200_at_50"

ACT_FUNC="leakyReLU"
COST_FUNC ="CE"

TEST_ROOT = './Data/fbank/'
TEST = '/fbank/test.ark'

PGRAM_ROOT ='./result/posteriorgram/'
PGRAM = MODEL + '.pgram'

MEM_DATA = 'data.fbank.memmap'
PKL_ID = 'ID.pkl'
MEM_DATA_shape = (621,1124823)
STATE_LENGTH = 1943
BATCH_SIZE = 1847
########################
#  load DNN open file  #
########################


layers,Ws,bs = pickle.load(open(MODEL_ROOT+MODEL,'rb')) 
nn = DNN(layers,Ws,bs,
         act=ACT_FUNC,
         cost=COST_FUNC)

# read Data #
mem_data = np.memmap(MEM_DATA,dtype='float32',mode='r',shape=MEM_DATA_shape)
IDs = readID(PKL_ID)
print "Data parsed"

########################
#  Save posteriorgram  #
########################

mem_shape = (STATE_LENGTH,len(IDs))
posteriorgram = np.memmap(PGRAM,dtype='float32',mode='w+',shape=mem_shape)

########################
#    Predict & To48    #
########################
print "Start test and saving..."
for idx in range(0,int(len(IDs)/BATCH_SIZE)):
    begin = idx * BATCH_SIZE
    end   = (idx+1) * BATCH_SIZE
    #if end > len(IDs):
    #    end = len(IDs)
    training_set = mem_data[:,begin:end]
    #training_set = mem_data[:,idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
    result = nn.test(training_set)
    # save result into pgram memmap.
    
    posteriorgram[:,begin:end] = result[:,:]

    del training_set
    del result
pdb.set_trace()
