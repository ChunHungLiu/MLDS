import cPickle as pickle

import numpy as np

import mywer
from reader import *
#from RNN.RNN_class import RNN_net
from RNN.RNN_class_GPU_NAG import RNN_net

# variables and parameters

DATA = "phonemeState"

BATCH_SIZE = 1

HIDDEN_LAYERS = [ 64 ]

START_LEARNING_RATE = LEARNING_RATE = 0.0002

MOMENTUM = 0.9
MOMENTUM_TYPE = 'rmsprop'

CLIP = 1

MAX_EPOCH = 300
SAVE_MODEL_EPOCH = 20

VAL_SET_RATIO = 1
L_RATE_DECAY_STEP = 10

MODEL_ROOT = "./rnn_result/model/"
MODEL = "_".join([ "DATA",DATA, \
                   "HIDDEN_LAYERS","-".join([ str(i) for i in HIDDEN_LAYERS ]), \
                   "L_RATE",str(START_LEARNING_RATE), \
                   "MOMENTUM",str(MOMENTUM), \
                   #"DROPOUT",str(DROPOUT),
                   "EPOCH",str(MAX_EPOCH) ])

PKL_ID = './ID.pkl'
#MEM_DATA = 'data.fbank.memmap'
PGRAM_ROOT= 'dnn_result/posteriorgram/'
DNN_MODEL = 'Angus_2'
MEM_PGRAM = PGRAM_ROOT+DNN_MODEL+'.pgram'
MEM_LABEL = 'label.memmap'
MEM_PGRAM_shape = (1124823,48)
STATE_LENGTH = 1943
PHONE_LENGTH = 48
#LABEL_VARIETY = 1943
LABEL_VARIETY = 48


print "Reading data..."
mem_pgram = np.memmap(MEM_PGRAM,dtype='float32',mode='r',shape=MEM_PGRAM_shape)
mem_label = np.memmap(MEM_LABEL,dtype='int16',mode='r',shape=(1124823,))
IDs = readID(PKL_ID)
idx = 0
IDs_utter = []
while idx <= len(IDs)-1:
    IDs_utter.append(["_".join(IDs[idx][0].split('_')[0:2]),IDs[idx][1]])
    #IDs_utter = [utter_name,utter_max]
    idx+=IDs[idx][1]

print "Preparing pickList..."
pickList = range(0,len(IDs_utter))
pickList = shuffle(pickList)
frame_max = max(IDs_utter, key=lambda x: x[1])
train_data_length = len(pickList)*VAL_SET_RATIO

DATA_LAYER  = [ mem_pgram.shape[1] ]
LABEL_LAYER = DATA_LAYER

LAYERS = DATA_LAYER + HIDDEN_LAYERS + LABEL_LAYER

########################
#     Create RNN       #
########################

print "Creating RNN..."
nn = RNN_net(LAYERS,
             batch_size=BATCH_SIZE,
             momentum_type="rmsprop",
             act_type="ReLU",
             cost_type="EU") 

########################
#      Train RNN       #
########################
val_label_vec = None
StateToVec = get_PhoneStateVec()
PhoneState = load_liststateto48()
PhoneIdx   = load_dict_IdxPh48()

p_s,p_e = 0,1
if p_e > len(pickList):
    p_e = len(pickList)

id_utter = 0
batched_inputs = []
batched_outputs = []
masks = []
mask_max = IDs_utter[id_utter][1]
utter_len = IDs_utter[id_utter][1]
start = sum(IDs_utter[i][1] for i in range(0,id_utter))
end   = start + utter_len

zeros = np.zeros((mask_max-utter_len, mem_pgram.shape[1]))

extended_in  = [ mem_pgram[start:end,:], zeros]

state_vec = [ (get_ph48_vec( PhoneIdx[PhoneState[i]] )) \
    for i in mem_label[start:end] ]
extended_out = [ state_vec,
                 np.zeros((mask_max-utter_len,
                           PHONE_LENGTH)
                 ) ]

utter_mask = [ np.ones((utter_len,1)),
               np.zeros((mask_max-utter_len,1)) ]

batched_inputs.append( np.vstack(extended_in) )
batched_outputs.append( np.vstack(extended_out) )
masks.append( np.vstack(utter_mask) )

x_seq = np.dstack(batched_inputs)
x_seq = np.asarray([ x.T for x in x_seq ],'float32')

#x_seq[:,0,:] = x_seq[:,0,:]/np.sum(x_seq,axis=2)

y_hat_seq = np.dstack(batched_outputs)
y_hat_seq = np.asarray([ y.T for y in y_hat_seq ],'float32')
mask_seq  = np.dstack(masks)
mask_seq  = np.asarray([ m.T for m in mask_seq ],'float32')

i_cost = np.sum((x_seq-y_hat_seq)**2)
print "init cost = {0}".format(i_cost)
totaltime = 0
print "Start training......"
for epoch in range(MAX_EPOCH):
    tStart = time.time()
    #cost = 0
 
    cost = nn.train(x_seq, y_hat_seq, mask_seq,
                    LEARNING_RATE,
                    MOMENTUM,
                    CLIP)

    tEnd = time.time()
    totaltime += tEnd - tStart

    #if (epoch+1 != MAX_EPOCH) and ((epoch+1) % L_RATE_DECAY_STEP == 0):
    #    print "learning rate annealed at epoch {0}".format(epoch+1)
    #    LEARNING_RATE*=0.9

    if epoch+1 != MAX_EPOCH and (epoch+1) % SAVE_MODEL_EPOCH == 0:
        fh = open(MODEL_ROOT+MODEL+"_at_{0}".format(epoch+1),'wb')
        saved_params = (nn.layers, nn.W, nn.Wh, nn.b)
        pickle.dump(saved_params, fh)
        fh.close()

    # Calculate Validation Set Error
    # TODO evaluate current Word error rate.
    valerror = None
    if VAL_SET_RATIO != 1:
        val_batch = BATCH_SIZE
        val_output = []
        total_count = 0
        val_error_count = 0
        for batch_num in range(int(train_data_length/val_batch),int(len(pickList)/val_batch)):
            p_s = batch_num * BATCH_SIZE
            p_e = (batch_num+1) * BATCH_SIZE
            if p_e > len(pickList):
                p_e = len(pickList)
	
            val_inputs = []
            val_outputs = []
            masks = []
            mask_max = max( IDs_utter[id_utter][1] \
                for id_utter in pickList[p_s:p_e] )
            for id_utter in pickList[ p_s:p_e ]:
                utter_len = IDs_utter[id_utter][1]
                start = sum(IDs_utter[i][1] for i in range(0,id_utter))
                end   = start + utter_len

                zeros = np.zeros((mask_max-utter_len, mem_pgram.shape[1]))
                extended_in  = [ mem_pgram[start:end,:], zeros]

                state_vec = [ [i] for i in mem_label[start:end] ]
                extended_out = [[0]]*(mask_max-utter_len)
                utter_mask = [ np.ones((utter_len,1)),
                           np.zeros((mask_max-utter_len,1)) ]

                val_inputs.append( np.vstack(extended_in) )
                val_outputs+=state_vec+extended_out
                masks.append( np.vstack(utter_mask) )

            x_seq = np.dstack(val_inputs)
            x_seq = np.asarray([ x.T for x in x_seq ],'float32')
            # TODO normalization to x_seq.
            y_hat_seq = np.hstack(val_outputs)
            mask_seq  = np.dstack(masks)
            mask_seq  = np.asarray([ m.T for m in mask_seq ],'float32')
            
            val_result = nn.test(x_seq,mask_seq)

            val_maxpositions = np.ravel(np.argmax(val_result,axis=2).T)
            total_count += val_maxpositions.shape[0]
            val_error_count += len([ i \
                for i,j in zip(val_maxpositions,y_hat_seq) if i!=j])
        valerror = float(val_error_count)/total_count

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

n_labels = 474
correct  = 0

y_seq = nn.test(x_seq, mask_seq)
y_seq = np.argmax(y_seq, axis=2)
answer = np.argmax(y_hat_seq, axis=2)

equal_entries = (y_seq == answer)
correct = np.sum( equal_entries )

correctness = 100 * ( correct / float(n_labels) )
print "Training set Result {0}".format(correctness) + "%"

########################
#      Save Model      #
########################

filehandler = open(MODEL_ROOT+MODEL,'wb')
saved_params = (nn.layers, nn.W, nn.Wh, nn.b)
pickle.dump(saved_params, filehandler)
filehandler.close()
