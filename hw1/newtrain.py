
import pdb
import pickle
import time

from nnet import *
from nnet.dnn import DNN
from nnet.dA import DA
from reader import *


########################
#   Define Paramters   #
########################
PHONEMAP = './Data/phones/48_39.map'

#TRAIN_FILENAME = "train_ant.ark"
#LABEL_FILENAME = "train_ant.lab"


TRAIN_FILENAME = "./Data/fbank/train.ark"
LABEL_FILENAME = "./Data/label/train.lab"

#TEST_FILENAME  = "train_ant.ark"

DATA = "fbank"
LABEL = "phoneme48"

BATCH_SIZE = 128

ERROR_FUNCTION ="Euclidean"

HIDDEN_LAYERS = [ 128 ]

START_LEARNING_RATE = LEARNING_RATE = 0.01

MOMENTUM = 0.0
RETAIN_PROB = 0.6
DROPOUT = 1-RETAIN_PROB

MAX_EPOCH = 100
SAVE_MODEL_EPOCH = 50

PRETRAIN_EPOCH = 100
L_RATE_DECAY_STEP = 101

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

# Get Training Set
print "Getting Training Data..."
TRAINING_SET = readData( TRAIN_FILENAME )
TRAINING_SET,VALIDATION_SET = splitset( TRAINING_SET )
TRAINING_ID,TRAINING_FEATURES = sepIDnFeat(TRAINING_SET)
VALIDATION_ID,VALIDATION_FEATURES = sepIDnFeat(VALIDATION_SET)

# Get Dictionaries
print "Gettting Dictionaries..."
ID2PHONE_DICT = readLabel( LABEL_FILENAME )
PHONE2INDEX_DICT = get_phone2index_dict(PHONEMAP,48)
ID2INDEX_DICT = get_ID2index_dict(ID2PHONE_DICT,PHONE2INDEX_DICT)

# Get Label
print "Getting Labels..."
LABEL_INDEXES = label(TRAINING_ID,ID2INDEX_DICT)
LABEL_FEATURES = index2vector(LABEL_INDEXES)

VALIDATION_INDEXES  = label(VALIDATION_ID,ID2INDEX_DICT)
VALIDATION_LABELS = index2vector(VALIDATION_INDEXES)

# Batch
print "Batching Training Data & Labels..."
BATCHED_TRAINING_FEATURES,BATCHED_LABEL_FEATURES = batch(TRAINING_FEATURES,LABEL_FEATURES,BATCH_SIZE)

# Transform to Numpy Column Vectors
print "Transform Batches to Numpy Column Vector Arrays..."
BATCHED_INPUT  = batchtoNpCol(BATCHED_TRAINING_FEATURES)
BATCHED_OUTPUT = batchtoNpCol(BATCHED_LABEL_FEATURES)
assert(len(BATCHED_INPUT) == len(BATCHED_OUTPUT))

# Remove unused variables
print "Removing Unused Variables..."
del TRAINING_SET,TRAINING_ID,TRAINING_FEATURES,LABEL_INDEXES,LABEL_FEATURES,\
    BATCHED_TRAINING_FEATURES,BATCHED_LABEL_FEATURES

#pdb.set_trace()
########################
#  Create Neural Net   #
########################
print "Constructing Neural Network..."
DATA_LAYER  = [ BATCHED_INPUT[0].shape[0] ]
LABEL_LAYER = [ BATCHED_OUTPUT[0].shape[0] ]
LAYERS = DATA_LAYER + HIDDEN_LAYERS + LABEL_LAYER

nn = DNN(LAYERS)

#pdb.set_trace()
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
# Initialize random seed

random.seed(time.time())

print "Preparing Validation Set"
VAL_INPUT = np.asarray(VALIDATION_FEATURES,dtype='float32')
VAL_ANSWER = VALIDATION_INDEXES
assert(VAL_INPUT.shape[0] == len(VAL_ANSWER))

VAL_NUM = len(VAL_ANSWER)
VAL_BATCHSIZE = max(int(VAL_NUM/100),1)
VAL_INPUT   = [  VAL_INPUT[i:i+VAL_BATCHSIZE] for i in range(0,VAL_NUM,VAL_BATCHSIZE)]
VAL_ANSWER  = [  VAL_ANSWER[i:i+VAL_BATCHSIZE] for ii in range(0,VAL_NUM,VAL_BATCHSIZE)]

#pdb.set_trace()

print "Start training......"
totaltime = 0
for epoch in range(MAX_EPOCH):
    tStart = time.time()
    cost = 0
    for batch_x,batch_y in zip(BATCHED_INPUT,BATCHED_OUTPUT):
        cost += nn.train(batch_x,batch_y,
                         LEARNING_RATE,
                         MOMENTUM)
    tEnd = time.time()
    totaltime += tEnd - tStart

    if (epoch+1 != MAX_EPOCH) and ((epoch+1) % L_RATE_DECAY_STEP == 0):
        print "learning rate annealed at epoch {0}".format(epoch+1)
        LEARNING_RATE /= 10

    if epoch+1 != MAX_EPOCH and (epoch+1) % SAVE_MODEL_EPOCH == 0:
        fh = open(MODEL_ROOT+MODEL+"_at_{0}".format(epoch+1),'wb')
        saved_params = (nn.layers, nn.W, nn.b)
        pickle.dump(saved_params, fh)
        fh.close()
    #pdb.set_trace()

    # Calculate Validation Set Error
    val_error_count = 0
    for val_x,val_ans in zip(VAL_INPUT,VAL_ANSWER):
        val_y = nn.test(val_x.T)
        val_max_idx = np.argmax(val_y,axis=0)
        val_error_count += len([i for i,j in zip(val_max_idx,val_ans) if i != j])

    val_error = float(val_error_count)/VAL_NUM


    print "Epoch:",epoch+1,"| Cost:",cost, "| Val Error", 100*val_error,'%', "| Epoch time:",tEnd-tStart

    # Reshuffle Batches
    TOTAL = zip(BATCHED_INPUT,BATCHED_OUTPUT)
    random.shuffle(TOTAL)
    BATCHED_INPUT = list(zip(*TOTAL))[0]
    BATCHED_OUTPUT = list(zip(*TOTAL))[1]

print "Total training time:",totaltime
########################
#  Training set Result #
########################
'''
result = nn.test(np.concatenate(BATCHED_INPUT,axis=1))
n_labels = result.shape[1]
result = np.argmax(result, axis=0)
answer = np.concatenate(BATCHED_OUTPUT,axis=1)
answer = np.argmax(answer, axis=0)
equal_entries = result == answer
correctness = np.sum( equal_entries ) / float(n_labels)
print "Traing set Result {0}".format(correctness)
'''
########################
#      Save Model      #
########################
filehandler = open(MODEL_ROOT+MODEL,'wb')
saved_params = (nn.layers, nn.W, nn.b)
pickle.dump(saved_params, filehandler)
filehandler.close()
