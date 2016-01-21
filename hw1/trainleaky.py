import datetime
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
#TRAIN_FILENAME = "train_ant.ark"
#LABEL_FILENAME = "train_ant.lab"

#TEST_FILENAME  = "train_ant.ark"

DATA = "fbank"
LABEL = "phonemeState"

BATCH_SIZE = 256
ACT_FUNC="leakyReLU"
COST_FUNC ="CE"

HIDDEN_LAYERS = [ 256,256 ] 

START_LEARNING_RATE = LEARNING_RATE = 0.001

MOMENTUM = 0.9
MOMENTUM_TYPE = 'adagrad'
MOMENTUM_TYPE = 'vanilla'
RETAIN_PROB = 0.9
DROPOUT = 1-RETAIN_PROB
INPUT_R_PROB = 0.9
MAX_NORM = 4

MAX_EPOCH = 500
SAVE_MODEL_EPOCH = 50

PRETRAIN_EPOCH = 0
L_RATE_DECAY_STEP = 1000

MODEL_ROOT = "./result/model/"
MODEL = "_".join([ "CE_DATA",DATA, \
                        "LABEL",LABEL, \
                        "HIDDEN_LAYERS","-".join([ str(i) for i in HIDDEN_LAYERS ]), \
                        "L_RATE",str(START_LEARNING_RATE), \
                        "MOMENTUM",str(MOMENTUM), \
                        "DROPOUT",str(DROPOUT),
                        "EPOCH",str(MAX_EPOCH) ])

########################
#  Read Data Build DNN #
########################
print "Reading data..."
LABELED_TRAINING_SET,LABELED_VALIDATION_SET = readfile()
print "Reading label..."
LABEL_DICT = readLabel() # TRAINING_LABEL is a dict()
print "Batching..."
LABELED_BATCHED_TRAINING_SET = batch( LABELED_TRAINING_SET,BATCH_SIZE )
print "Matching label..."
LABELED_BATCHED_LABEL = MatchLabel2Batches( LABELED_BATCHED_TRAINING_SET,LABEL_DICT )
print "Removing frame name..."
BATCHED_TRAINING_SET   = removeBatchLabel(LABELED_BATCHED_TRAINING_SET)
BATCHED_LABEL = removeBatchLabel(LABELED_BATCHED_LABEL)
print "Transforming into vectors..."
BATCHED_VECTORS = BatchedLabelToVector(BATCHED_LABEL)
print "Transforming into NP..."
BATCHED_INPUT = BatchToNPCol(BATCHED_TRAINING_SET)
BATCHED_OUTPUT = BatchToNPCol(BATCHED_VECTORS)

DATA_LAYER  = [ len( BATCHED_TRAINING_SET[0][0] ) ]
LABEL_LAYER = [ len( BATCHED_VECTORS[0][0] ) ]

# pdb.set_trace()
LAYERS = DATA_LAYER + HIDDEN_LAYERS + LABEL_LAYER

print "Data parsed!!!"

########################
#  Create Neural Net   #
########################
model_path = "./result/model/DATA_fbank_LABEL_phoneme48_HIDDEN_LAYERS_1024-1024_L_RATE_0.01_MOMENTUM_0.9_DROPOUT_0.1_EPOCH_500_at_100" 
ll,ww,bb = pickle.load(open(model_path,'rb'))
nn = DNN(ll,
         ww,
         bb,
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
print "Preparing Validation Set"

PhoneState = load_liststateto39()

Val_IDs, Val_Feats = SepIDnFeat(LABELED_VALIDATION_SET)

val_x = np.asarray(Val_Feats,dtype='float32').T
val_label = [ LABEL_DICT[ID] for ID in Val_IDs ]

# phone48to39 = load_dict_48to39()

# pdb.set_trace()

totaltime = 0
print "Start training......"
for epoch in range(100,MAX_EPOCH):
    tStart = time.time()
    cost = 0
    for batched_inputs,batched_outputs in zip(BATCHED_INPUT,BATCHED_OUTPUT):
        cost += nn.train(batched_inputs,batched_outputs,
                         LEARNING_RATE,
                         MOMENTUM,
                         RETAIN_PROB,
                         INPUT_R_PROB)
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
    # Calculate Validation Set Error
    val_batch = 500
    val_output = []
    for i in xrange( (val_x.shape[1]-1)/val_batch +1):
        begin = i*val_batch
        if (i+1)*val_batch > val_x.shape[1]:
            end = val_x.shape[1]
        else:
            end = (i+1)*val_batch
        val_y = nn.test(val_x[:,begin:end])
        val_maxpositions = np.argmax(val_y,axis=0)
        val_output += [ PhoneState[pos] for pos in val_maxpositions ]

    val_error_count = len([ i for i,j in zip(val_output,val_label) if i != j])
    valerror = float(val_error_count)/len(val_output)

    tStartR = time.time()
    print "Epoch:",epoch+1,"| Cost:",cost, "| Val Error", 100*valerror,'%', "| Epoch time:",tEnd-tStart

    print "Reshuffling..."
    LABELED_BATCHED_TRAINING_SET = batch( LABELED_TRAINING_SET,BATCH_SIZE )
    LABELED_BATCHED_LABEL = MatchLabel2Batches( LABELED_BATCHED_TRAINING_SET,LABEL_DICT )
    BATCHED_TRAINING_SET   = removeBatchLabel(LABELED_BATCHED_TRAINING_SET)
    BATCHED_LABEL = removeBatchLabel(LABELED_BATCHED_LABEL)
    BATCHED_VECTORS = BatchedLabelToVector(BATCHED_LABEL)
    BATCHED_INPUT = BatchToNPCol(BATCHED_TRAINING_SET)
    BATCHED_OUTPUT = BatchToNPCol(BATCHED_VECTORS)
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
