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

BATCH_SIZE = 128

ERROR_FUNCTION ="euclidean"

HIDDEN_LAYERS = [ 2048 ] * 3

START_LEARNING_RATE = LEARNING_RATE = 0.01

MOMENTUM = 0.9
RETAIN_PROB = 0.9
DROPOUT = 1-RETAIN_PROB

MAX_EPOCH = 200
SAVE_MODEL_EPOCH = 50

PRETRAIN_EPOCH = 100
L_RATE_DECAY_STEP = 101

VAL_SET_RATIO = 1

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
print "Reading data and label..."
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
pdb.set_trace()
print "Data parsed!!!"

########################
#  Create Neural Net   #
########################

nn = DNN(LAYERS)

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
'''
print "Preparing Validation Set"

PhoneState = load_liststateto39()

StateToVec = get_PhoneStateVec()
val_label_vec = [ StateToVec[ID] for ID in val_label ]
'''
# phone48to39 = load_dict_48to39()

# pdb.set_trace()

totaltime = 0
val_batch = 128
val_output = []
print "Start training......"
for epoch in range(MAX_EPOCH):
    tStart = time.time()
    cost = 0
    for batch_num in range(0,int(len(pickList)/BATCH_SIZE)):
        start = batch_num*BATCH_SIZE
        batched_inputs = mem_data[:,start]
        batched_outputs = StateToVec[mem_label[start]]
        for idx in pickList[batch_num*BATCH_SIZE+1:(batch_num+1)*BATCH_SIZE]:
            batched_inputs = np.column_stack((batched_inputs,mem_data[:,idx]))
            batched_outputs = np.column_stack((batched_outputs,StateToVec[mem_label[idx]]))
        cost += nn.train(batched_inputs,batched_outputs,
                         LEARNING_RATE,
                         MOMENTUM,
                         RETAIN_PROB,
                         1)
    tEnd = time.time()
    totaltime += tEnd - tStart
    if (epoch+1 != MAX_EPOCH) and ((epoch+1) % L_RATE_DECAY_STEP == 0):
        print "learning rate annealed at epoch {0}".format(epoch+1)
        LEARNING_RATE*=0.9
    
    if epoch+1 != MAX_EPOCH and (epoch+1) % SAVE_MODEL_EPOCH == 0:
        fh = open(MODEL_ROOT+MODEL+"_at_{0}".format(epoch+1),'wb')
        saved_params = (nn.layers, nn.W, nn.b)
        pickle.dump(saved_params, fh)
        fh.close()
    # Calculate Validation Set Error
    # move the declaration outside the loop
    '''
    for i in xrange( (val_data.shape[1]-1)/val_batch +1):
        begin = i*val_batch
        if (i+1)*val_batch > val_data.shape[1]:
            end = val_data.shape[1]
        else:
            end = (i+1)*val_batch
        val_y = nn.test(val_data[:,begin:end])
        val_maxpositions = np.argmax(val_y,axis=0)
        val_output += [ PhoneState[pos] for pos in val_maxpositions ]

    val_error_count = len([ i for i,j in zip(val_output,val_label_vec) if i != j])
    valerror = float(val_error_count)/len(val_output)
    '''
    tStartR = time.time()
    print "Epoch:",epoch+1,"| Cost:",cost, "| Val Error", 100*1,'%', "| Epoch time:",tEnd-tStart

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
