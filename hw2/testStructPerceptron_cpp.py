import csv
import pickle
import pdb
import os
import sys

from reader import *
from struct_perceptron import *

########################
#   Define Paramters   #
########################

MODEL_ROOT = "./struct_perceptron/model/"
MODEL = "cpp_1_at_10"

TEST_ROOT = './Data/fbank/'
#TEST = '/fbank/test.ark'
TEST = 'test.ark'

FIX = True

PREDICTION_ROOT ='./struct_perceptron/phone_sequence/'
PREDICTION = MODEL + '.csv'

BATCH_SIZE = 1
PKL_ID = './ID_test.pkl'
PGRAM_ROOT= 'dnn_result/posteriorgram/'
DNN_MODEL = 'Angus_2'
MEM_PGRAM = PGRAM_ROOT+DNN_MODEL+'_test.pgram'
MEM_PGRAM_shape = (180406,48)

########################
#  load SP open file   #
########################

print "Loading StructPerceptron..."
saved_w = pickle.load(open(MODEL_ROOT+MODEL,'rb')) 
#sp = StructPerceptron(48,48) 
#sp.w[:] = saved_w

#IDs,TEST_DATA,VAL_DATA = readfile_for_test( TEST_ROOT+TEST,1 )

print "Reading data..."
mem_pgram = np.memmap(MEM_PGRAM,dtype='float32',mode='r',shape=MEM_PGRAM_shape)
IDs = readID(PKL_ID)
idx = 0
IDs_utter = []
while idx <= len(IDs)-1:
    IDs_utter.append(["_".join(IDs[idx][0].split('_')[0:2]),IDs[idx][1]])
    #IDs_utter = [utter_name,utter_max]
    idx+=IDs[idx][1]

print "Preparing pickList..."
pickList = range(0,len(IDs_utter))
#pickList = shuffle(pickList)
frame_max = max(IDs_utter, key=lambda x: x[1])

########################
#       Predict        #
########################

# Get Dictionaries
PhoneMapIdxtoPh48 = load_liststateto48()
PhoneMap48to39 = load_dict_48to39()
PhoneMap39toChr = load_dict_48toChr()
id_phSeq_pair = []

for batch_num in range(0,int(len(pickList)/BATCH_SIZE)):
    sys.stdout.write("\repoch {0} : data {1} of {2}".format(1, batch_num, int(len(pickList)/BATCH_SIZE)))
    sys.stdout.flush()
    p_s = batch_num * BATCH_SIZE
    p_e = (batch_num+1) * BATCH_SIZE
    if p_e >= len(pickList):
        p_e = len(pickList)
	
    batched_inputs = []
    mask_max = max( IDs_utter[id_utter][1] \
        for id_utter in pickList[p_s:p_e] )
    for id_utter in pickList[ p_s:p_e ]:
        utter_len = IDs_utter[id_utter][1]
        start = sum(IDs_utter[i][1] for i in range(0,id_utter))
        end   = start + utter_len

        zeros = np.zeros((mask_max-utter_len, mem_pgram.shape[1]))
        extended_in  = [ i for i in mem_pgram[start:end,:], zeros]

        batched_inputs.append( np.vstack(extended_in) )

    x_seq = np.dstack(batched_inputs)
    x_seq = np.asarray([ x.T for x in x_seq ],'float32')
    shape = x_seq[:,0,:].shape

    # write args to file
    with open("test_cpp.tmp", 'w') as arg_f:
        # x_dim y_dim
        arg_f.write("48\n48\n")
        # w
        arg_f.write(' '.join(str(i) for i in saved_w))
        arg_f.write('\n')
        # x_len
        arg_f.write('{0}'.format(shape[0]))      
        arg_f.write('\n')
        # x
        list_like = x_seq[:,0,:].reshape((mask_max * 48,1))
        arg_f.write(' '.join(str(i[0]) for i in list_like))
        arg_f.write('\n')
    
    cmd = ("./test_struct_perceptron test_cpp.tmp")
    response = os.popen(cmd).read()
    response = response.split()
    y_seq = [int(e) for e in response]
    #pdb.set_trace()
    
    for one_seq, utter_idx in zip([y_seq], pickList[p_s:p_e]) :
        utter_len = IDs_utter[utter_idx][1]
        
        one_seq = [ PhoneMap48to39[ PhoneMapIdxtoPh48[ph]] \
            for ph in one_seq[:utter_len] ]
        final_seq = []
        if FIX:
            # open a window of size 3, a|b|c 
            # if a == c and b != a, then directly change b to a :P
            final_seq.append( one_seq[0] )
            for i in xrange(1,len(one_seq)-1):
                if one_seq[i-1] == one_seq[i+1] and \
                   one_seq[i]   != one_seq[i-1]:
                    final_seq.append( one_seq[i-1] )
                elif one_seq[i] != one_seq[i-1] and \
                     one_seq[i] != one_seq[i+1]:
                    # discard
                    continue
                    #final_seq.append( one_seq[i-1] )
                else:
                    final_seq.append( one_seq[i] )
            final_seq.append( one_seq[-1] ) 
            #assert len(one_seq) == len(final_seq)

            one_seq = final_seq
        pruned_seq = []
        prev = None
        for l in xrange(len(one_seq)):
            if prev != one_seq[l]:
                pruned_seq.append( one_seq[l] )
            prev = one_seq[l]
        if pruned_seq[0] == 'sil':
            del pruned_seq[0]
        if len(pruned_seq) == 0:
            pdb.set_trace()
        if pruned_seq[-1] == 'sil':
            del pruned_seq[-1]
        
        phone_seq = [ PhoneMap39toChr[ph] \
            for ph in pruned_seq ]
        phone_seq = ''.join(phone_seq)
        #pdb.set_trace()
        utter_id = IDs_utter[utter_idx][0]

        id_phSeq_pair.append( (utter_id, phone_seq) )

########################
#      Save CSV        #
########################
PRED_FILE = open( PREDICTION_ROOT + PREDICTION ,'wb')
HEADER = ["id","phone_sequence"]

c = csv.writer(PRED_FILE,delimiter =',')
c.writerow(HEADER)
c.writerows(id_phSeq_pair)

PRED_FILE.close()
