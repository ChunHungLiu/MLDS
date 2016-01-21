import cPickle as pickle
import os
import sys

import numpy as np

import mywer
from struct_perceptron import *
from reader import *


# variables and parameters
DATA = "phoneme48"

BATCH_SIZE = 1

MAX_EPOCH = 100
SAVE_MODEL_EPOCH = 10

VAL_SET_RATIO = 1

MODEL_ROOT = "./struct_perceptron/model/"
MODEL = "cpp_1"

PKL_ID = './ID.pkl'
#MEM_DATA = 'data.fbank.memmap'
PGRAM_ROOT= 'dnn_result/posteriorgram/'
DNN_MODEL = 'Angus_2'
MEM_PGRAM = PGRAM_ROOT+DNN_MODEL+'.pgram'
MEM_LABEL = 'label48.memmap'
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

########################
#  Create Perceptron   #
########################

#sp = StructPerceptron(48,48) 

########################
#   Train Perceptron   #
########################
val_label_vec = None
#StateToVec = get_PhoneStateVec()
# label is 48 ph idx.

# 48 ph idx -> 48 ph chr
PhoneState = load_liststateto48()
'''
# 48 ph chr -> 39 ph chr
#PhoneMap48to39 = load_dict_48to39()
'''
# 39 ph chr -> 39 ph Idx
PhoneIdx   = load_dict_IdxPh48()



epoch = 0
totaltime = 0
totalupdates = 0
keep_training = True
weight_vec = []
for i in xrange(48*(2*48+2)):
  weight_vec.append(0)

print "Start training......"
while keep_training and epoch < MAX_EPOCH:
    tStart = time.time()
    cost = 0
    no_mistake = True
    for batch_num in range(0,int(train_data_length/BATCH_SIZE)):
        sys.stdout.write("\repoch {0} : data {1} of {2}".format(epoch+1, batch_num, int(train_data_length/BATCH_SIZE)))
        sys.stdout.flush()
        p_s = batch_num * BATCH_SIZE
        p_e = (batch_num+1) * BATCH_SIZE
        if p_e > len(pickList):
            p_e = len(pickList)

        batched_inputs = []
        batched_outputs = []
        mask_max = max( IDs_utter[id_utter][1] \
            for id_utter in pickList[p_s:p_e] )
        for id_utter in pickList[ p_s:p_e ]:
            utter_len = IDs_utter[id_utter][1]
            start = sum(IDs_utter[i][1] for i in range(0,id_utter))
            end   = start + utter_len

            zeros = np.zeros((mask_max-utter_len, mem_pgram.shape[1]))
            extended_in  = [ i for i in mem_pgram[start:end,:], zeros]

            state_idx = [ [i] \
                for i in mem_label[start:end] ]
            extended_out = [ state_idx,
                             np.zeros((mask_max-utter_len,
                                       1)
                             ) ]
            batched_inputs.append( np.vstack(extended_in) )
            batched_outputs.append( np.vstack(extended_out) )

        x_seq = np.dstack(batched_inputs)
        x_seq = np.asarray([ x.T for x in x_seq ],'float32')
        y_hat_seq = np.dstack(batched_outputs)
        y_hat_seq = np.asarray([ y.T for y in y_hat_seq ],'float32')
        shape = x_seq[:,0,:].shape

        # write args to file
        with open("args.tmp", 'w') as arg_f:
          # x_dim y_dim
          arg_f.write("48\n48\n")
          # w
          arg_f.write(' '.join(str(i) for i in weight_vec))
          arg_f.write('\n')
          # x_len
          arg_f.write('{0}'.format(shape[0]))
          arg_f.write('\n')
          # x
          list_like = x_seq[:,0,:].reshape((mask_max * 48,1))
          arg_f.write(' '.join(str(i[0]) for i in list_like))
          arg_f.write('\n')
          # y
          arg_f.write(' '.join(str(i) for i in y_hat_seq[:,0,0]))

        cmd = ("./struct_perceptron_cpp args.tmp")
        response = os.popen(cmd).read()
        response = response.split()

        if response[0] == '1':
            no_mistake = False
            weight_vec = [float(e) for e in response[1:]]
            #pdb.set_trace()
            np_weight = np.array(weight_vec)
            np_weight /= np.sqrt(np.sum(np_weight**2))
            weight_vec = np_weight.tolist()

            totalupdates += 1

    if no_mistake:
        keep_training = False
    tEnd = time.time()
    totaltime += tEnd - tStart

    # Calculate Validation Set Error
    # TODO evaluate current Word error rate.
    '''
    valerror = None
    if VAL_SET_RATIO != 1:
        val_batch = BATCH_SIZE
        total_count = 0
        val_error_count = 0
        for batch_num in range(int(train_data_length/val_batch),int(len(pickList)/val_batch)):
            p_s = batch_num * BATCH_SIZE
            p_e = (batch_num+1) * BATCH_SIZE
            if p_e > len(pickList):
                p_e = len(pickList)

            val_inputs = []
            val_outputs = []
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

                val_inputs.append( np.vstack(extended_in) )
                val_outputs.append(state_vec+extended_out)

            x_seq = np.dstack(val_inputs)
            x_seq = np.asarray([ x.T for x in x_seq ],'float32')
            y_hat_seq = np.dstack(val_outputs)

            val_result, _ = sp.get_best(x_seq[:,0,:])
            #pdb.set_trace()
            nequal = (np.argmax(val_result, axis=2) != y_hat_seq[:,0,:])
            total_count += nequal.shape[0] * nequal.shape[1]
            val_error_count += np.sum(nequal)
        valerror = float(val_error_count)/total_count
    '''

    tStartR = time.time()
    if VAL_SET_RATIO != 1:
        print "Epoch:",epoch+1,"| Cost:",cost,"| Val Error:", 100*valerror,'%', "| Epoch time:",tEnd-tStart
    else:
        print "Epoch:",epoch+1,"| Cost:",cost,"| Epoch time:",tEnd-tStart
    
    if epoch+1 != MAX_EPOCH and (epoch+1) % SAVE_MODEL_EPOCH == 0:
        print "   Saving model..."
        fh = open(MODEL_ROOT+MODEL+"_at_{0}".format(epoch+1),'wb')
        saved_params = (weight_vec)
        pickle.dump(saved_params, fh)
        fh.close()
    print "Reshuffling..."
    pickList = shuffle(pickList)
    tEndR = time.time()
    print "Reshuffle time {0}".format(tEndR-tStartR)

    epoch += 1

print totaltime
print "total updates : {0}".format(totalupdates)
########################
#  Traing set Result   #
########################
'''
n_labels = 0
correct  = 0
for batch_num in range(0,int(len(pickList)/BATCH_SIZE)):
    p_s = batch_num * BATCH_SIZE
    p_e = (batch_num+1) * BATCH_SIZE
    if p_e > len(pickList):
        p_e = len(pickList)

    batched_inputs = []
    batched_outputs = []
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

        batched_inputs.append( np.vstack(extended_in) )
        batched_outputs.append(state_vec+extended_out)

    x_seq = np.dstack(batched_inputs)
    x_seq = np.asarray([ x.T for x in x_seq ],'float32')
    answer = np.dstack(batched_outputs)

    y_seq,_ = sp.get_best(x_seq)
    #y_seq = np.argmax(y_seq, axis=2)
    #answer = [ PhoneIdx[PhoneState[state_idx] ] \
    #    for state_idx in mem_label[ pickList[p_s:p_e] ] ]

    equal_entries = (y_seq == answer[:,0,:])
    n_labels += equal_entries.shape[0] * equal_entries.shape[1]
    correct += np.sum( equal_entries )

correctness = 100 * ( correct / float(n_labels) )
print "Training set Result {0}".format(correctness) + "%"
'''
########################
#      Save Model      #
########################

filehandler = open(MODEL_ROOT+MODEL,'wb')
saved_params = (weight_vec)
pickle.dump(saved_params, filehandler)
filehandler.close()
