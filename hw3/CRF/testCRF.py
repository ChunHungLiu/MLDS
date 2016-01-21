import cPickle
import csv
import numpy as np
import pdb
import time

from Readfile import *
from Viterbi import Viterbi

global y_class

TEST_ROOT = 'test/'
TEST_NAME = 'EJ_shuffle60.csv'
y_class = 48

def Psi (x , y):
    psi_o = np.zeros( (y_class,len(x[0])) )
    psi_t = np.zeros( (y_class+1,y_class+1) )
    for idx in range(len(x)):
        psi_o[y[idx]] += x[idx]
        if idx == 0 :
            psi_t[y_class][y[idx]] += 1
        elif idx == len(x)-1:
            psi_t[y[idx-1]][y[idx]] += 1
            psi_t[y[idx]][y_class] += 1
        else:
            psi_t[y[idx-1]][y[idx]] += 1
    return np.hstack( ( np.hstack(psi_o) , np.hstack(psi_t)  )  )

with open('model/EJ_Per_normalize_160',"rb") as filehandler:
    w = cPickle.load(filehandler)
filehandler.close()

def load_liststateto48():
    liststateto48 = []
    with open('state_48_39.map','r') as fin:
        for line in fin:
            liststateto48.append(line.split()[1])
    return liststateto48

def load_dict_48to39():
    d = dict()
    with open('48_39.map') as fin:
        for row in fin:
            pair = row.split()
            d[ pair[0] ] = pair[1]
    
    if len(set(d.keys())) != 48 or len(set(d.values())) != 39:
        print "Error phoneme amount!"
    return d

def load_dict_48toChr():
    d = dict()
    with open('48_idx_chr-5.map') as fin:
        for line in fin:
            ph48, idx, char= line.split()
            d[ ph48 ] = char
    return d


PhoneMapIdxtoPh48 = load_liststateto48()
PhoneMap48to39 = load_dict_48to39()
PhoneMap39toChr = load_dict_48toChr()

xs,IDs_utter = read_test()

idNphrase=[]
for idx in xrange(0,len(xs)):
    x = xs[idx]
    id_utter = IDs_utter[idx][0]
    y_hat = [0]*len(x)
    V = Viterbi (x , w , y_class , y_hat , 0)
    start = time.clock()
    y_tilde = V.main_Viterbi()
    end = time.clock()
    print "Viterbi time :" , end-start
    print "tilde " , np.dot(w.T , Psi( x , y_tilde ))
    y_temp = [PhoneMap48to39[ PhoneMapIdxtoPh48[int(ph)]] for ph in y_tilde]
    smooth_y = []
    smooth_y1 = []
    smooth_y1.append(y_temp[0])
    for i in xrange(1,len(y_temp)-1):
        if y_temp[i-1] == y_temp[i+1] and \
           y_temp[i]   != y_temp[i-1]:
            smooth_y1.append( y_temp[i-1] )
        elif y_temp[i] != y_temp[i-1] and \
             y_temp[i] != y_temp[i+1]:
            # discard
            continue
            #final_seq.append( y_temp[i-1] )
        else:
            smooth_y1.append( y_temp[i] )
    smooth_y.append(smooth_y1[0])
    for i in xrange(1,len(smooth_y1)-1):
        if smooth_y1[i-1] == smooth_y1[i+1] and \
           smooth_y1[i]   != smooth_y1[i-1]:
            smooth_y.append( smooth_y1[i-1] )
        elif smooth_y1[i] != smooth_y1[i-1] and \
             smooth_y1[i] != smooth_y1[i+1]:
            # discard
            continue
            #final_seq.append( y_temp[i-1] )
        else:
            smooth_y.append( smooth_y1[i] )
    pruned_y = []
    prev = None
    for l in xrange(len(smooth_y)):
        if prev != smooth_y[l]:
            pruned_y.append( smooth_y[l] )
        prev = smooth_y[l]
    while pruned_y[0] == 'sil':
        del pruned_y[0]
    while pruned_y[-1] == 'sil':
        del pruned_y[-1]
    if len(pruned_y) == 0:
        pdb.set_trace()
    y_tosave = ''.join([PhoneMap39toChr[ph] for ph in pruned_y])
    print [id_utter,y_tosave]
    idNphrase.append([id_utter,y_tosave])


PRED_FILE = open( TEST_ROOT + TEST_NAME ,'wb')
HEADER = ["id","phone_sequence"]

c = csv.writer(PRED_FILE,delimiter =',')
c.writerow(HEADER)
c.writerows(idNphrase)

PRED_FILE.close() 
