import csv
import pdb
import pickle
import time

from nnet import *
from nnet.dnn import DNN
from nnet.dA import dA
from reader import *


########################
#   Define Paramters   #
########################
TRAIN_FILENAME = "train_ant.ark"
LABEL_FILENAME = "train_ant.lab"

#TRAIN_FILENAME = "./Data/fbank/train.ark"
#LABEL_FILENAME = "./Data/label/train.lab"

#TEST_FILENAME  = "train_ant.ark"

DATA_TYPE = "fbank"
LABEL_TYPE = "phoneme48"

BATCH_SIZE = 128

ERROR_FUNCTION ="euclidean"

HIDDEN_LAYERS = [ 128 ]

#MOMENTUM = 0.5

#DROPUT = 0.5

#LEARNING_RATE = 0.01

MAX_EPOCH = 1000
PRE_TRAIN_EPOCH = 100

########################
#  Read Data Build DNN #
########################
print "Reading Data..."

LABELED_TRAINING_SET,LABELED_VALIDATION_SET = readfile( TRAIN_FILENAME )

LABEL_DICT = readLabel( LABEL_FILENAME ) # TRAINING_LABEL is a dict()

LABELED_BATCHED_TRAINING_SET = batch( LABELED_TRAINING_SET,BATCH_SIZE )

LABELED_BATCHED_LABEL = MatchLabel2Batches( LABELED_BATCHED_TRAINING_SET,LABEL_DICT )

BATCHED_TRAINING_SET   = removeBatchLabel(LABELED_BATCHED_TRAINING_SET)

BATCHED_LABEL = removeBatchLabel(LABELED_BATCHED_LABEL)
BATCHED_VECTORS = BatchedLabelToVector(BATCHED_LABEL)

BATCHED_INPUT = BatchToNPCol(BATCHED_TRAINING_SET)
BATCHED_OUTPUT = BatchToNPCol(BATCHED_VECTORS)


validationNlabel = []

for data in LABELED_VALIDATION_SET:
    validationNlabel.append(data + [LABEL_DICT[data[0]]])


DATA_LAYER  = [ len( BATCHED_TRAINING_SET[0][0] ) ]
LABEL_LAYER = [ len( BATCHED_VECTORS[0][0] ) ]

# pdb.set_trace()
LAYERS = DATA_LAYER + HIDDEN_LAYERS + LABEL_LAYER

########################
#  Create Neural Net   #
########################
print "Initializing Neural Network..."

nn = DNN(LAYERS)

########################
# pre-Train Neural Net #
########################
'''
prop_input = _DATA_BATCH_
for l in xrange(1,len(nn.layers)):
    for epoch in PRE_TRAIN_EPOCH:
        nn.pretrainer[l].train()

    # get the hiiden vector of current dA and
    # pass into the next dA as data.
    prop_input = nn.pretrainer[l].get_hidden(prop_input)

'''
########################
#   Train Neural Net   #
########################
print "Start training"

# phone48to39 = load_dict_48to39()

# pdb.set_trace()

p48list = load_list39to48()

p48to39dict = load_dict_48to39()

totaltime = 0
for epoch in range(MAX_EPOCH):
    tStart = time.time()
    cost = 0
    for batched_inputs,batched_outputs in zip(BATCHED_INPUT,BATCHED_OUTPUT):
        cost += nn.train(batched_inputs,batched_outputs)
    tEnd = time.time()
    totaltime += tEnd - tStart

    # Calculate Validation Error
    valsum = 0
    for val in validationNlabel:
        val_x = np.transpose( np.asarray([val[1]],dtype='float32') )
        p_feat = nn.test(val_x)
        pos = np.argmax(p_feat)
        p_48 = p48list[pos]
        p_39 = p48to39dict[p_48]
        if val[2] == p_48:
            valsum += 1
        print "valdiating:",val[2],p_48
    valcorrect = float(valsum)/len(validationNlabel)

    print "Epoch:",epoch,"Cost:",cost, "Epoch time:",tEnd-tStart,"Val correct,",valcorrect
print totaltime



########################
#      Save Model      #
########################
'''
MODEL_ROOT = "./result/model/"
MODEL = "baseline"

#MODEL = "_".join([ DATA_TYPE, \
#                        LABEL_TYPE, \
#                        "HIDDEN_LAYERS","-".join([ str(i) for i in HIDDEN_LAYERS ]), \
#                        MAX_EPOCH \
#                        ])

filehandler = open(MODEL_ROOT+MODEL,'wb')
pickle.dump(nn,filehandler)
filehandler.close()
'''

########################
#   Define Paramters   #
########################

MODEL_ROOT = "./result/model/"
MODEL = "baseline"

#TEST_ROOT = './Data'
#TEST = '/fbank/test.ark'

PREDICTION_ROOT ='./result/prediction/'
PREDICTION = MODEL + '.csv'

########################
#  load DNN open file  #
########################

# nn = pickle.load( open(MODEL_ROOT+MODEL,'rb') )

#TEST_DATA,VAL_DATA = readfile(TRAIN_FILENAME,1)

TEST_DATA,VAL_DATA = readfile( TEST_ROOT + TEST,1 )
#PRED_FILE = open( PREDICTION_ROOT + PREDICTION ,'wb')

# Get Dictionaries
Phone48 = load_list39to48()
PhoneMap48to39 = load_dict_48to39()

# For CSV
HEADER = ["Id","Prediction"]

########################
#       Predict        #
########################
print "Start predicting..."

#c = csv.writer(PRED_FILE,delimiter =',')
#c.writerow(HEADER)

predictions = []
print PhoneMap48to39

for data in TEST_DATA:
    # p_feat is a vector
    x = np.transpose( np.asarray([data[1]],dtype='float32') )
    p_feat = nn.test(x)

    # Get index of element closest to Value
    # Value = 1
    # pos = min(enumerate(p_feat), key=lambda x:abs(x[1]-Value))[0]
    pos =  np.argmax(p_feat)
    # pos = p_feat.index(max(p_feat))

    phone48 = Phone48[pos]
    phone39 = PhoneMap48to39[phone48]

    pred = [ data[0],phone39 ]
    print pred
    predictions.append(pred)

#c.writerows(predictions)


#PRED_FILE.close()
