import cPickle as pickle

import numpy as np

import mywer
from reader import *
#from RNN.RNN_class import RNN_net
from RNN.RNN_class_GPU_NAG import RNN_net

# variables and parameters

DATA = "phoneme48"

BATCH_SIZE = 1

HIDDEN_LAYERS = [ 128 ]

START_LEARNING_RATE = LEARNING_RATE = 0.0001
LEARNING_RATE_ANNEALED = 0.2

MOMENTUM = 0.9
RMS_ALPHA = 0.9
CLIP = 1

MOMENTUM_TYPE = "rms+NAG"
ACT_FUNC = "ReLU"
COST_FUNC = "EU"

MAX_EPOCH = 120
SAVE_MODEL_EPOCH = 20

VAL_SET_RATIO = 0.8

MODEL_ROOT = "./rnn_result/model/"
MODEL = "_".join([ "LAYERS","-".join([ str(i) for i in HIDDEN_LAYERS ]), \
                   "L_RATE",str(START_LEARNING_RATE), \
                   "MOMEN",str(MOMENTUM_TYPE), \
                   "COST",str(COST_FUNC),\
                   #"DROPOUT",str(DROPOUT),
                   "L_RATE_A",str(LEARNING_RATE_ANNEALED) ])
MODEL = 'SGD_6'
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
pdb.set_trace()
DATA_LAYER  = [ mem_pgram.shape[1] ]
LABEL_LAYER = DATA_LAYER

LAYERS = DATA_LAYER + HIDDEN_LAYERS + LABEL_LAYER

########################
#     Create RNN       #
########################

print "Creating RNN..."
nn = RNN_net(LAYERS,
             batch_size = BATCH_SIZE,
             momentum_type = MOMENTUM_TYPE,
             act_type = ACT_FUNC,
             cost_type = COST_FUNC)

########################
#      Train RNN       #
########################
val_label_vec = None
StateToVec = get_PhoneStateVec()
PhoneState = load_liststateto48()
PhoneIdx   = load_dict_IdxPh48()

prev_err = float('inf')
prev_2 = float('inf')
prev_3 = float('inf')
cal_dev = 3
totaltime = 0
print "Start training......"
print "BATCH_SIZE:",BATCH_SIZE,",HIDDEN_LAYERS:",HIDDEN_LAYERS,",L_RATE:",LEARNING_RATE,",MOMENTUM_TYPE:",MOMENTUM_TYPE,",ACT_FUNC:",ACT_FUNC,"COST_FUNC:",COST_FUNC
for epoch in range(MAX_EPOCH):
    tStart = time.time()
    flag = 0
    cost = 0
    for batch_num in range(0,int(train_data_length/BATCH_SIZE)):
        p_s = batch_num * BATCH_SIZE
        p_e = (batch_num+1) * BATCH_SIZE
        if p_e > len(pickList):
            p_e = len(pickList)

        batched_inputs = []
        batched_outputs = []
        masks = []
        mask_max = max( IDs_utter[id_utter][1] \
            for id_utter in pickList[p_s:p_e] )
        for id_utter in pickList[ p_s:p_e ]:
            utter_len = IDs_utter[id_utter][1]
            start = sum(IDs_utter[i][1] for i in range(0,id_utter))
            end   = start + utter_len

            zeros = np.zeros((mask_max-utter_len, mem_pgram.shape[1]))
            extended_in  = [ i for i in mem_pgram[start:end,:], zeros]

            state_vec = [ get_ph48_vec( PhoneIdx[PhoneState[i]] ) \
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
        '''
        for dim in range(0,len(batched_inputs)):
            for vec in range(0,batched_inputs[0].shape[1]):
                batched_inputs[vec][dim] = (batched_inputs[vec][dim]-0.5)*2
        '''
        x_seq = np.dstack(batched_inputs)
        x_seq = np.asarray([ x.T for x in x_seq ],'float32')
        y_hat_seq = np.dstack(batched_outputs)
        y_hat_seq = np.asarray([ y.T for y in y_hat_seq ],'float32')
        mask_seq  = np.dstack(masks)
        mask_seq  = np.asarray([ m.T for m in mask_seq ],'float32')

        cost += nn.train(x_seq, y_hat_seq, mask_seq,
                         LEARNING_RATE,
                         RMS_ALPHA,
                         CLIP,
                         MOMENTUM)

    tEnd = time.time()
    totaltime += tEnd - tStart

    err_range = float('inf')
    if cal_dev == 0:
        #sec_m = (prev_3**2 + prev_2**2 + prev_err**2) / 3
        m = [prev_3, prev_2, prev_err]
        err_range = max(m) - min(m)
        print "err_range : ", err_range
    if cal_dev > 0:
        cal_dev -= 1

    if (epoch+1 != MAX_EPOCH) and (err_range < 0.2): #epoch+1 != 1) and (prev_cost<cost):
        #LEARNING_RATE*=LEARNING_RATE_ANNEALED
        flag = 1
        cal_dev = 3

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
                extended_in  = [ i for i in mem_pgram[start:end,:], zeros]

                state_vec = [ [PhoneIdx[PhoneState[i] ] ] for i in mem_label[start:end] ]
                extended_out = [[0]]*(mask_max-utter_len)
                utter_mask = [ np.ones((utter_len,1)),
                           np.zeros((mask_max-utter_len,1)) ]

                val_inputs.append( np.vstack(extended_in) )
                val_outputs.append(state_vec+extended_out)
                masks.append( np.vstack(utter_mask) )

            x_seq = np.dstack(val_inputs)
            x_seq = np.asarray([ x.T for x in x_seq ],'float32')
            y_hat_seq = np.dstack(val_outputs)
            mask_seq  = np.dstack(masks)
            mask_seq  = np.asarray([ m.T for m in mask_seq ],'float32')

            val_result = nn.test(x_seq,mask_seq)

            nequal = (np.argmax(val_result, axis=2) != y_hat_seq[:,0,:])
            total_count += nequal.shape[0] * nequal.shape[1]
            val_error_count += np.sum(nequal)
            #pdb.set_trace()
            #val_maxpositions = np.ravel(np.argmax(val_result,axis=2).T)
            #total_count += val_maxpositions.shape[0]
            #val_error_count += len([ i \
            #    for i,j in zip(val_maxpositions,y_hat_seq) if i!=j])
        valerror = float(val_error_count)/total_count

    tStartR = time.time()
    if VAL_SET_RATIO != 1:
        prev_3, prev_2, prev_err = prev_2, prev_err, 100*valerror
        print "Epoch:",epoch+1,"| Cost:",cost,"| Val Error:", 100*valerror,'%', "| Epoch time:",tEnd-tStart
    else:
        print "Epoch:",epoch+1,"| Cost:",cost,"| Epoch time:",tEnd-tStart

    if epoch+1 != MAX_EPOCH and (epoch+1) % SAVE_MODEL_EPOCH == 0:
        print "   Saving model..."
        fh = open(MODEL_ROOT+MODEL+"_at_{0}".format(epoch+1)+",l_rate:{0}".format(valerror),'wb')
        saved_params = (nn.layers, nn.W, nn.Wh, nn.b)
        pickle.dump(saved_params, fh)
        fh.close()
    print "Reshuffling..."
    pickList = shuffle(pickList)
    tEndR = time.time()
    print "Reshuffle time {0}".format(tEndR-tStartR)

    if flag==1:
        print "!!!  Learning rate annealed to:",LEARNING_RATE,"  !!!"

print totaltime

########################
#  Traing set Result   #
########################

n_labels = 0
correct  = 0
for batch_num in range(0,int(len(pickList)/BATCH_SIZE)):
    p_s = batch_num * BATCH_SIZE
    p_e = (batch_num+1) * BATCH_SIZE
    if p_e > len(pickList):
        p_e = len(pickList)

    batched_inputs = []
    batched_outputs = []
    masks = []
    mask_max = max( IDs_utter[id_utter][1] \
        for id_utter in pickList[p_s:p_e] )
    for id_utter in pickList[ p_s:p_e ]:
        utter_len = IDs_utter[id_utter][1]
        start = sum(IDs_utter[i][1] for i in range(0,id_utter))
        end   = start + utter_len

        zeros = np.zeros((mask_max-utter_len, mem_pgram.shape[1]))
        extended_in  = [ i for i in mem_pgram[start:end,:], zeros]

        state_vec = [ [PhoneIdx[PhoneState[i] ] ] \
            for i in mem_label[start:end] ]
        extended_out = [[0]]*(mask_max-utter_len)

        utter_mask = [ np.ones((utter_len,1)),
                       np.zeros((mask_max-utter_len,1)) ]

        batched_inputs.append( np.vstack(extended_in) )
        batched_outputs.append(state_vec+extended_out)
        masks.append( np.vstack(utter_mask) )

    x_seq = np.dstack(batched_inputs)
    x_seq = np.asarray([ x.T for x in x_seq ],'float32')
    answer = np.dstack(batched_outputs)
    mask_seq  = np.dstack(masks)
    mask_seq  = np.asarray([ m.T for m in mask_seq ],'float32')

    y_seq = nn.test(x_seq, mask_seq)
    y_seq = np.argmax(y_seq, axis=2)
    #answer = [ PhoneIdx[PhoneState[state_idx] ] \
    #    for state_idx in mem_label[ pickList[p_s:p_e] ] ]

    equal_entries = (y_seq == answer[:,0,:])
    n_labels += equal_entries.shape[0] * equal_entries.shape[1]
    correct += np.sum( equal_entries )

correctness = 100 * ( correct / float(n_labels) )
print "Training set Result {0}".format(correctness) + "%"
########################
#      Save Model      #
########################

filehandler = open(MODEL_ROOT+MODEL,'wb')
saved_params = (nn.layers, nn.W, nn.Wh, nn.b)
pickle.dump(saved_params, filehandler)
filehandler.close()

'''
# test open file.
layers,Ws,Whs,bs = pickle.load(open(MODEL_ROOT+MODEL,'rb'))
nn = RNN_net(layers,Ws,Whs,bs,
             batch_size=BATCH_SIZE,
             momentum_type=MOMENTUM_TYPE,
             act_type=ACT_FUNC,
             cost_type=COST_FUNC)
n_labels = 0
correct  = 0
for batch_num in range(0,int(len(pickList)/BATCH_SIZE)):
    p_s = batch_num * BATCH_SIZE
    p_e = (batch_num+1) * BATCH_SIZE
    if p_e > len(pickList):
        p_e = len(pickList)

    batched_inputs = []
    batched_outputs = []
    masks = []
    mask_max = max( IDs_utter[id_utter][1] \
        for id_utter in pickList[p_s:p_e] )
    for id_utter in pickList[ p_s:p_e ]:
        utter_len = IDs_utter[id_utter][1]
        start = sum(IDs_utter[i][1] for i in range(0,id_utter))
        end   = start + utter_len

        zeros = np.zeros((mask_max-utter_len, mem_pgram.shape[1]))
        extended_in  = [ mem_pgram[start:end,:], zeros]

        state_vec = [ [PhoneIdx[PhoneState[i] ] ] \
            for i in mem_label[start:end] ]
        extended_out = [[0]]*(mask_max-utter_len)

        utter_mask = [ np.ones((utter_len,1)),
                       np.zeros((mask_max-utter_len,1)) ]

        batched_inputs.append( np.vstack(extended_in) )
        batched_outputs.append(state_vec+extended_out)
        masks.append( np.vstack(utter_mask) )

    x_seq = np.dstack(batched_inputs)
    x_seq = np.asarray([ x.T for x in x_seq ],'float32')
    answer = np.dstack(batched_outputs)
    mask_seq  = np.dstack(masks)
    mask_seq  = np.asarray([ m.T for m in mask_seq ],'float32')

    y_seq = nn.test(x_seq, mask_seq)
    y_seq = np.argmax(y_seq, axis=2)
    #answer = [ PhoneIdx[PhoneState[state_idx] ] \
    #    for state_idx in mem_label[ pickList[p_s:p_e] ] ]

    equal_entries = (y_seq == answer[:,0,:])
    n_labels += equal_entries.shape[0] * equal_entries.shape[1]
    correct += np.sum( equal_entries )

correctness = 100 * ( correct / float(n_labels) )
print "Training set Result {0}".format(correctness) + "%"
'''
