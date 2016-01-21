import csv
import pickle
import pdb

from reader import *
from RNN.RNN_class_GPU_NAG import RNN_net

########################
#   Define Paramters   #
########################

MODEL_ROOT = "./rnn_result/model/"
MODEL = "SGD_3"

TEST_ROOT = './Data/fbank/'
#TEST = '/fbank/test.ark'
TEST = 'test.ark'

MOMENTUM_TYPE="rms+NAG"
ACT_FUNC="ReLU"
COST_FUNC="EU"

PGRAM_ROOT ='./dnn_result/posteriorgram/'
PGRAM = MODEL + '.pgram'

BATCH_SIZE = 1
PKL_ID = './ID_test.pkl'
PGRAM_ROOT= 'dnn_result/posteriorgram/'
DNN_MODEL = 'Angus_2'
MEM_PGRAM = PGRAM_ROOT+DNN_MODEL+'_test.pgram'
MEM_PGRAM_shape = (180406,48)

########################
#  load RNN open file  #
########################

print "Loading RNN..."
layers,Ws,Whs,bs = pickle.load(open(MODEL_ROOT+MODEL,'rb')) 
nn = RNN_net(layers,Ws,Whs,bs,
             batch_size=BATCH_SIZE,
             momentum_type=MOMENTUM_TYPE,
             act_type=ACT_FUNC,
             cost_type=COST_FUNC)

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

###############################################
mem_shape = (len(IDs),48)
posteriorgram = np.memmap(PGRAM_ROOT+PGRAM,
                          dtype='float32',
                          mode='w+',
                          shape=mem_shape)
########################
#       Predict        #
########################

# Get Dictionaries
PhoneIdx   = load_dict_IdxPh48()

for batch_num in range(0,int(len(pickList)/BATCH_SIZE)):
    p_s = batch_num * BATCH_SIZE
    p_e = (batch_num+1) * BATCH_SIZE
    if p_e >= len(pickList):
        p_e = len(pickList)
	
    batched_inputs = []
    masks = []
    mask_max = max( IDs_utter[id_utter][1] \
        for id_utter in pickList[p_s:p_e] )
    for id_utter in pickList[ p_s:p_e ]:
        utter_len = IDs_utter[id_utter][1]
        start = sum(IDs_utter[i][1] for i in range(0,id_utter))
        end   = start + utter_len

        zeros = np.zeros((mask_max-utter_len, mem_pgram.shape[1]))
        extended_in  = [ i for i in mem_pgram[start:end,:], zeros]

        utter_mask = [ np.ones((utter_len,1)),
                       np.zeros((mask_max-utter_len,1)) ]

        batched_inputs.append( np.vstack(extended_in) )
        masks.append( np.vstack(utter_mask) )

    x_seq = np.dstack(batched_inputs)
    x_seq = np.asarray([ x.T for x in x_seq ],'float32')
    mask_seq  = np.dstack(masks)
    mask_seq  = np.asarray([ m.T for m in mask_seq ],'float32')

    result = nn.test(x_seq, mask_seq)
    result = result.reshape(end-start,48)
    posteriorgram[start:end] += result[:]

    #posteriorgram[start:end,:] /= phone_map_freq

    # normalize to porb.
    #ph_sum = np.zeros((BATCH_SIZE,1))
    #ph_sum[:,0] = np.sum(posteriorgram[begin:end,:], axis=1)[:]
    #posteriorgram[start:end,:] /= ph_sum
    
    print result
    #posteriorgram[begin:end,:] = result[:,:].T
    del result
