import cPickle
import csv
import pdb
import time

import numpy as np
import theano
import theano.tensor as T

from reader import *

class HMM(object):
    def __init__(self,state_num,observe_num,min_duration = 3, max_duration = 11):
        self.state_num = state_num * max_duration

        self.min_duration = min_duration
        self.max_duration = max_duration

        # Initial Probability
        self.init_prob  = np.zeros( self.state_num )
        # State to State Transition Probability
        self.trans_prob = np.zeros(( self.state_num, self.state_num ))
        # State to Observation Probability
        self.result = []

    def read(self,pgram,label,IDs,IDs_utter):
        MAX = self.max_duration
        MIN = self.min_duration

        data_length = pgram.shape[0]
        assert data_length == len(label)

        def divide_by_col(x1,x2):
            return np.transpose( np.transpose(x1) / x2 )

        def split_by_utter(label,IDs_utter):
            utter_label = []
            prev_idx = 0
            cur_idx  = 0
            for utter in IDs_utter:
                cur_idx += utter[1]
                utter_label.append(label[prev_idx:cur_idx])
                prev_idx = cur_idx
            return utter_label

        # Calculate Phone Number
        phone_count = np.zeros(self.state_num)

        # Get count of phonemes in data
        for utter_label in split_by_utter(label, IDs_utter):
            count = 0
            for idx in range(len(utter_label)):
                s = utter_label[idx]

                phone_count[ s * MAX + count ] += 1

                if idx < len(utter_label) - 1:
                    s_prime = utter_label[idx+1]
                    if s_prime == s:
                        if count + 1 < MAX:
                            count += 1
                    else:
                        count = 0

        try:
            assert np.sum(phone_count) == len(IDs) # - len(IDs_utter)
        except AssertionError:
            print "Sum of phone_count is", np.sum(phone_count)

        for i in range(len(phone_count)):
            if not phone_count[i]:
                phone_count[i] = 1.0

        # Find Initial Probability
        for utter_label in split_by_utter(label,IDs_utter):
            self.init_prob[ utter_label[0] * MAX ] += 1
        self.init_prob /= len(IDs_utter)

        try:
            assert np.sum(self.init_prob) == 1.0
        except AssertionError:
            print "Sum of initial probability is ",np.sum(self.init_prob)

        # Find Transition Probability
        for utter_label in split_by_utter(label,IDs_utter):
            count = 0
            for idx in range(len(utter_label)):
                s = utter_label[idx]

                if idx == len(utter_label) - 1:
                    self.trans_prob[ s * MAX + count ][ s * MAX + count ] += 1
                else:
                    s_prime = utter_label[idx+1]

                    if s_prime == s:
                        if count + 1 == MAX:
                            self.trans_prob[ s * MAX + count ][ s_prime * MAX + count ] += 1
                        else:
                            self.trans_prob[ s * MAX + count ][ s_prime * MAX + count + 1 ] += 1
                            count += 1
                    else:
                        self.trans_prob[ s * MAX + count ][ s_prime * MAX ] += 1
                        count = 0

        self.trans_prob = divide_by_col(self.trans_prob, phone_count)

        for s in range(self.state_num):
            try:
                prob_sum = np.sum(self.trans_prob[s,:])
                assert prob_sum == 1 or prob_sum == 0
            except AssertionError:
                print "Sum of transition probability is", np.sum(self.trans_prob[s,:])

    def train(self,pgram, IDs, IDs_utter):
        pass

    def test(self, pgram, IDs, IDs_utter):

        def split_by_utter(pgram,IDs_utter):
            utter_pgram = []
            prev_idx = 0
            cur_idx  = 0
            for utter in IDs_utter:
                cur_idx += utter[1]
                utter_pgram.append(pgram[prev_idx:cur_idx])
                prev_idx = cur_idx
            return utter_pgram

        def viterbi(utter):
            MAX = self.max_duration
            MIN = self.min_duration

            states     = []
            V          = np.zeros((self.state_num,len(utter)))
            prev_state = np.zeros((self.state_num,len(utter)))

            def trace_prev_states(p, l):
                for t in range(MIN-1):
                    if int(p/MAX) != int(prev_state[p][l]/MAX):
                        return False
                    p = prev_state[p][l]
                    l -= 1
                return True

            for l in range(len(utter)):
                if not l:
                    for s in range(self.state_num):
                        V[s][l] = utter[l][int(s/MAX)]
                else:
                    for s_prime in range(self.state_num):
                        residue = s_prime % MAX
                        if MAX <= MIN:
                            if not residue or ( residue == MAX - 1 and trace_prev_states( s_prime, l - 1 ) ):
                                # Calculate all possibilities
                                candidates = [ V[ s ][ l-1 ] * \
                                                self.trans_prob[ s ][ s_prime ] * \
                                                utter[ l ][ int(s_prime/MAX) ] \
                                                for s in range( self.state_num ) ]
                                V[ s_prime ][ l ] = max( candidates )
                                prev_state[ s_prime ][ l ] = candidates.index( max( candidates ) )
                            else:
                                s = s_prime - 1
                                V[ s_prime ][ l ] = V[ s ][ l - 1 ] * \
                                                    self.trans_prob[ s ][ s_prime ] * \
                                                    utter[ l ][ int(s_prime/MAX) ]
                                prev_state[ s_prime ][ l ] = s
                        else:
                            if not residue or residue >= MIN:
                                # Calculate all possibilities
                                candidates = [ V[ s ][ l-1 ] * \
                                                self.trans_prob[ s ][ s_prime ] * \
                                                utter[ l ][ int(s_prime/MAX) ] \
                                                for s in range( self.state_num ) ]
                                V[ s_prime ][ l ] = max( candidates )
                                prev_state[ s_prime ][ l ] = candidates.index( max( candidates ) )
                            else:
                                s = s_prime - 1
                                V[ s_prime ][ l ] = V[ s ][ l - 1 ] * \
                                                    self.trans_prob[ s ][ s_prime ] * \
                                                    utter[ l ][ int(s_prime/MAX) ]
                                prev_state[ s_prime ][ l ] = s
                        """
                        if l < MIN or not trace_prev_states( s_prime, l-1 ):
                            s = s_prime
                            # forward calculate
                            V[ s_prime ][ l ] = \
                            V[ s ][ l - 1 ] * self.trans_prob[ s ][ s_prime ] * utter[ l ][ int(s_prime/MAX) ]
                            prev_state[ s_prime ][ l ] = s
                        else: # Can switch phonemes
                            candidates = [ V[ s ][ l-1 ] * self.trans_prob[ s ][ s_prime ] * utter[ l ][ int(s_prime/MAX) ] \
                                            for s in range( self.state_num ) ]
                            V[ s_prime ][ l ] = max( candidates )
                            prev_state[ s_prime ][ l ] = candidates.index( max( candidates ) )
                        """

            # Get Viterbi Results
            cur_state = np.argmax( V[:,len(utter)-1] )
            for track in reversed(range(len(utter))):
                states.insert(0,cur_state)
                cur_state = prev_state[cur_state][track]

            return states

        for utter in split_by_utter(pgram, IDs_utter):
            tStart = time.time()
            self.result.append( viterbi(utter) )
            print "{0}/{1} of testing data, Time taken: {2} seconds".format(len(self.result),len(IDs_utter),time.time()-tStart)

def get_training_data():
    PKL_ID = './ID.pkl'
    #MEM_DATA = 'data.fbank.memmap'
    PGRAM_ROOT= 'dnn_result/posteriorgram/'
    DNN_MODEL = 'Angus_2'
    MEM_PGRAM = PGRAM_ROOT+DNN_MODEL+'.pgram'
    MEM_LABEL = 'label48.memmap'
    MEM_PGRAM_shape = (1124823,48)
    PHONE_LENGTH = 48
    LABEL_VARIETY = 48

    mem_pgram = np.memmap(MEM_PGRAM,dtype='float32',mode='r',shape=MEM_PGRAM_shape)
    mem_label = np.memmap(MEM_LABEL,dtype='int16',mode='r',shape=(1124823,))

    IDs = readID(PKL_ID)
    idx = 0
    IDs_utter = []
    while idx <= len(IDs)-1:
        IDs_utter.append(["_".join(IDs[idx][0].split('_')[0:2]),IDs[idx][1]])
        idx+=IDs[idx][1]
    return mem_pgram, mem_label, IDs, IDs_utter

def get_testing_data():
    PKL_ID = './ID_test.pkl'
    PGRAM_ROOT= 'dnn_result/posteriorgram/'
    DNN_MODEL = 'Angus_2'
    MEM_PGRAM = PGRAM_ROOT + DNN_MODEL + '_test.pgram'
    MEM_PGRAM_shape = (180406,48)
    mem_pgram = np.memmap(MEM_PGRAM, dtype='float32',mode='r',shape=MEM_PGRAM_shape)
    IDs = readID(PKL_ID)
    idx = 0
    IDs_utter = []
    while idx <= len(IDs)-1:
        IDs_utter.append(["_".join(IDs[idx][0].split('_')[0:2]),IDs[idx][1]])
        idx+=IDs[idx][1]
    return mem_pgram, IDs, IDs_utter

########################
#      Save CSV        #
########################
def save(IDs_utter, result,min_duration, max_duration):
    def map2seq(sequence):
        def MapSeqTo39(seq):
            PhoneMapIdxtoPh48 = load_liststateto48()
            PhoneMap48to39 = load_dict_48to39()
            return [ PhoneMap48to39[ PhoneMapIdxtoPh48[int(s/max_duration)] ] for s in seq ]

        def fix_seq(seq):
            # Duration Model
            for idx in range(1,len(seq)-1):
                if seq[idx-1] == seq[idx+1] and seq[idx] != seq[idx-1]:
                    seq[idx] = seq[idx-1]
            # Contract the Sequence
            seq = [ s for idx, s in enumerate(seq) if idx != 0 and seq[idx] != seq[idx-1] ]
            # Remove silence
            if not all( s == 'sil' for s in seq ):
                while seq[0] == 'sil':
                    seq.pop(0)
                while seq[-1] == 'sil':
                    seq.pop(-1)
            return seq

        PhoneMap39toChr = load_dict_48toChr()

        return ''.join([ PhoneMap39toChr[s] for s in fix_seq(MapSeqTo39(seq) ) ])

    MODEL = "_".join(['Hmm','hsmm',str(min_duration),str(max_duration)])
    PREDICTION_ROOT =''
    PREDICTION = 'output.kaggle'
    PRED_FILE = open( PREDICTION_ROOT + PREDICTION ,'wb')
    HEADER = ["id","phone_sequence"]

    # Map to phone sequence
    id_seq_pair = [ (ID[0],map2seq(seq)) for ID, seq in zip(IDs_utter,result) ]

    c = csv.writer(PRED_FILE,delimiter =',')
    c.writerow(HEADER)
    c.writerows(id_seq_pair)

    PRED_FILE.close()

def main():
    print "Reading Data..."
    train_mem_pgram, train_mem_label, train_IDs, train_IDs_utter = get_training_data()
    test_mem_pgram, test_IDs, test_IDs_utter = get_testing_data()

    print "Initializing HMM..."
    state_num = obsv_num = train_mem_pgram.shape[1]
    hmm = HMM(state_num,obsv_num)

    print "HMM is counting probability..."
    hmm.read( train_mem_pgram, train_mem_label, train_IDs, train_IDs_utter)
    #pdb.set_trace()

    print "HMM is now testing..."
    hmm.test( test_mem_pgram, test_IDs, test_IDs_utter )

    save(test_IDs_utter,hmm.result, hmm.min_duration, hmm.max_duration)



if __name__ == "__main__":
    main()
