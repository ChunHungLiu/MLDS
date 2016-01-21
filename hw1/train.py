import datetime
import pdb
import pickle
import time
import numpy as np

from nnet import *
from nnet.dnn import DNN
from nnet.dA import DA
from reader import *


########################
#   Define Paramters   #
########################
#TRAIN_FILENAME = "train_ant.ark"
#LABEL_FILENAME = "train_ant.lab"

PKL_ID = 'ID.pkl'
MEM_DATA = 'data.fbank.memmap'
MEM_LABEL = 'label.memmap'
MEM_DATA_shape = (621,1124823)
LABEL_VARIETY = 1943

#TEST_FILENAME  = "train_ant.ark"

DATA = "fbank"
LABEL = "phonemeState"

BATCH_SIZE = 256
ACT_FUNC="leakyReLU"
COST_FUNC ="CE"

HIDDEN_LAYERS = [ 2048 ] * 3

START_LEARNING_RATE = LEARNING_RATE = 0.05

MOMENTUM = 0.9
MOMENTUM_TYPE = 'rms+NAG'
RMS_ALPHA = 0.9
RETAIN_PROB = 0.9
DROPOUT = 1-RETAIN_PROB
INPUT_R_PROB = 0.9
MAX_NORM = 4

MAX_EPOCH = 360
SAVE_MODEL_EPOCH = 40

PRETRAIN_EPOCH = 100
L_RATE_DECAY_STEP = 101

#VAL_SET_RATIO = 1
VAL_SET_RATIO = 0.95

MODEL_ROOT = "./result/model/"
MODEL = "_".join([ "DATA",DATA, \
                   "LABEL",LABEL, \
                   "HIDDEN_LAYERS","-".join([ str(i) for i in HIDDEN_LAYERS ]), \
                   "L_RATE",str(START_LEARNING_RATE), \
                   "MOMENTUM",str(MOMENTUM), \
                   "DROPOUT",str(DROPOUT),
                   "EPOCH",str(MAX_EPOCH) ])

########################
#  Read Data Build DNN #
########################
print "Read data and label..."
mem_data = np.memmap(MEM_DATA,dtype='float32',mode='r',shape=MEM_DATA_shape )
mem_label = np.memmap(MEM_LABEL,dtype='int16',mode='r')
IDs = readID(PKL_ID)
print "Preparing pickList & val_set..."
pickList = [num for num in range(0,len(IDs))]
pickList = shuffle(pickList)
pickList,val_data,val_label,val_IDs = parse_val_set(mem_data,mem_label,pickList,IDs,VAL_SET_RATIO)
# TRAINING_LABEL is a dict()

DATA_LAYER  = [ mem_data.shape[0] ]
LABEL_LAYER = [ LABEL_VARIETY ]

# pdb.set_trace()
LAYERS = DATA_LAYER + HIDDEN_LAYERS + LABEL_LAYER

print "Data parsed!!!"

########################
#  Create Neural Net   #
########################

nn = DNN(LAYERS,
         m_norm=MAX_NORM,
         act=ACT_FUNC,
         cost=COST_FUNC,
         momentum_type=MOMENTUM_TYPE)

########################
# pre-Train Neural Net #
########################
'''
print "Start pre-training. pretrain {0} epoches".format(PRETRAIN_EPOCH)
prop_input = data
for l,da in enumerate(DAs):
    for epoch in xrange(PRETRAIN_EPOCH):
        batch_cost = 0
        tStart = time.time()
        for i in xrange( (data.shape[1]-1)/PRETRAIN_BATCH_SIZE + 1):
            begin = i * PRETRAIN_BATCH_SIZE
            if (i+1)*PRETRAIN_BATCH_SIZE > data.shape[1]:
                end = data.shape[1]
            else:
                end = (i+1)*PRETRAIN_BATCH_SIZE
            b = da.train(prop_input[:,begin:end])
            if b != b:
                pdb.set_trace()
            batch_cost += b
        tEnd = time.time()
        print "Layer:{0}, Epoch:{1}, cost:{2}, time:{3}".format(
            l,epoch+1,batch_cost,tEnd-tStart)
    prop_input = da.get_hidden(prop_input)

'''
########################
#   Train Neural Net   #
########################
val_label_vec = None
StateToVec = get_PhoneStateVec()

if VAL_SET_RATIO != 1:
    print "Preparing Validation Set"
    PhoneState = load_liststateto39()
    val_label_vec = np.vstack([ StateToVec[ID] for ID in val_label ]).T
# phone48to39 = load_dict_48to39()

totaltime = 0
print "Start training......"
for epoch in range(MAX_EPOCH):
    tStart = time.time()
    cost = 0
    for batch_num in range(0,int(len(pickList)/BATCH_SIZE)):
        start = batch_num*BATCH_SIZE    
        end   = (batch_num+1)*BATCH_SIZE
        if end > len(pickList):
            end = len(pickList)
	batched_inputs = mem_data[:,pickList[start:end]]
        batched_outputs = \
            np.vstack( [ StateToVec[i] for i in mem_label[pickList[start:end] ]] ).T

        cost += nn.train(batched_inputs,batched_outputs,
                         LEARNING_RATE,
                         MOMENTUM,
                         RMS_ALPHA,
                         RETAIN_PROB,
                         INPUT_R_PROB)

    tEnd = time.time()
    totaltime += tEnd - tStart
    print "Now: BATCH_SIZE=",BATCH_SIZE,"Learning_rate=",LEARNING_RATE
    if epoch%360==0 and epoch !=0:
        LEARNING_RATE*=0.9
    if epoch == 100:
        BATCH_SIZE=128
    
    if epoch+1 != MAX_EPOCH and (epoch+1) % SAVE_MODEL_EPOCH == 0:
        fh = open(MODEL_ROOT+MODEL+"_at_{0}".format(epoch+1),'wb')
        saved_params = (nn.layers, nn.W, nn.b)
        pickle.dump(saved_params, fh)
        fh.close()
    # Calculate Validation Set Error
    # move the declaration outside the loop
    valerror = None
    if VAL_SET_RATIO != 1:
        val_batch = 512
        val_output = []
        val_error_count = 0
        for i in xrange( (val_data.shape[1]-1)/val_batch +1):
            begin = i*val_batch
            end = (i+1)*val_batch
            if end > val_data.shape[1]:
                end = val_data.shape[1]

            val_result = nn.test(val_data[:,begin:end])
            val_maxpositions = np.argmax(val_result,axis=0)
            #pdb.set_trace()
            #val_output += [ PhoneState[pos] for pos in val_maxpositions ]
            #val_output += val_maxpositions.tolist()
            val_error_count += len([ i \
                for i,j in zip(val_maxpositions,val_label[begin:end]) if i!=j])
        valerror = float(val_error_count)/len(val_label)

    tStartR = time.time()
    if VAL_SET_RATIO != 1:
        print "Epoch:",epoch+1,"| Cost:",cost,"| Val Error:", 100*valerror,'%', "| Epoch time:",tEnd-tStart
    else:
        print "Epoch:",epoch+1,"| Cost:",cost,"| Epoch time:",tEnd-tStart

    print "Reshuffling..."
    pickList = shuffle(pickList)
    tEndR = time.time()
    print "Reshuffle time {0}".format(tEndR-tStartR)


print totaltime

########################
#  Traing set Result   #
########################

n_labels = 0
correct  = 0
for batched_inputs,batched_outputs in zip(BATCHED_INPUT,BATCHED_OUTPUT):
    result = nn.test(batched_inputs)
    n_labels += result.shape[1]
    result = np.argmax(result, axis=0)
    answer = np.argmax(batched_outputs, axis=0)
    equal_entries = result == answer
    correct += np.sum( equal_entries )
correctness = 100 * ( correct / float(n_labels) )
print "Training set Result {0}".format(correctness) + "%"

########################
#      Save Model      #
########################

filehandler = open(MODEL_ROOT+MODEL,'wb')
saved_params = (nn.layers, nn.W, nn.b)
pickle.dump(saved_params, filehandler)
filehandler.close()

