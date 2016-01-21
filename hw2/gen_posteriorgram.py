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

MODEL_ROOT = "./dnn_result/model/"
MODEL = "Angus_2"

ACT_FUNC="leakyReLU"
COST_FUNC ="CE"

TEST_ROOT = './Data/fbank/'
TEST = 'test.ark'

PGRAM_ROOT ='./dnn_result/posteriorgram/'
PGRAM = MODEL + '.pgram'

MEM_DATA = './data.fbank.memmap'
PKL_ID = './ID.pkl'
MEM_DATA_shape = (621,1124823)
STATE_LENGTH = 1943
PHONE_LENGTH = 48
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

mem_shape = (len(IDs),PHONE_LENGTH)
posteriorgram = np.memmap(PGRAM_ROOT+PGRAM,
                          dtype='float32',
                          mode='w+',
                          shape=mem_shape)

########################
#    Predict & To48    #
########################

PhoneState = load_liststateto48()
PhoneIdx   = load_dict_IdxPh48()

phone_map_freq = np.zeros(PHONE_LENGTH)
for i in xrange(STATE_LENGTH):
    ph48_idx = PhoneIdx[ PhoneState[i] ]
    phone_map_freq[ph48_idx] += 1

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
    result = result.T
    for i in xrange(STATE_LENGTH):
        ph48_idx = PhoneIdx[ PhoneState[i] ]
        posteriorgram[begin:end, ph48_idx] += result[:,i]

    posteriorgram[begin:end,:] /= phone_map_freq

    # normalize to porb.
    ph_sum = np.zeros((BATCH_SIZE,1))
    ph_sum[:,0] = np.sum(posteriorgram[begin:end,:], axis=1)[:]
    posteriorgram[begin:end,:] /= ph_sum
    
    print posteriorgram[begin:end,:]
    #posteriorgram[begin:end,:] = result[:,:].T
    del training_set
    del result
