import cPickle
import csv
import numpy as np
import pdb
import time
import random

from Readfile import *
from Viterbi import Viterbi
from math import exp

global y_class

MODEL_ROOT = 'model/'
MODEL_NAME = 'CRF_test1'
TEST_ROOT = 'test/'
TEST_NAME = 'CRF_test1.csv'

ITERATION = 20

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
    psi_t = psi_t / len(x) 
    return np.hstack( ( np.hstack(psi_o) , np.hstack(psi_t)  )  )
def update_w(w , x , y_hat , y_tilde , V):
    start = time.clock() 
    #sig = V.update_Viterbi()
    end = time.clock()
    print "update  time :" , end-start
    #sig = V.CRF_Psi[len(x)+1][0] / V.CRF_Prob[len(x)+1][0]
    #print "sig = " , sig
    #print "sig = "  , sig
    #print "Psi = "  , Psi (x,y_tilde) 
    #print "exp  = " , V.CRF_Psi[len(x)+1][0]
    #print "Z(x) = " , V.CRF_Prob[len(x)+1][0]
    #y_diff = y_hat - y_tilde
    n = len([t for h,t in zip(y_hat,y_tilde) if h!=t])
    print "n= ",n                                        
    #print "P(y_hat|x) = " , exp( np.dot( w , Psi(x,y_hat) ) ) / V.CRF_Prob[len(x)+1][0] 
    #w = w + 1 * ( Psi( x , y_hat ) * ( 1 + 1 ) - sig  )
    w = w + 1 * ( (1+4*n/len(x))*Psi( x , y_hat ) - (4*n/len(x))*Psi( x , y_tilde) )   
    #w = w + 1 * ( 1.1 * Psi( x , y_hat ) - 0.1 * Psi(x,y_tilde) - 1 * sig )#Psi( x , y_tilde) )   
    #w = w / np.linalg.norm(w)
    return w 
print "Start .........."

# initial 
 
#[x , y_hat] = gen_data.gen_data()
xs , y_hats  = read_examples()

w_o = np.zeros( ( y_class , len(xs[0][0])) )
w_t = np.zeros( ( y_class+1 ,y_class+1) )
w = np.hstack( ( np.hstack(w_o) , np.hstack(w_t)  )  )

for train_num in range(ITERATION):
    print "###########################   now on iter:",train_num+1,"   ############################"
    temp_w = np.copy(w)
    pickList = list(range(0,len(y_hats)))
    random.shuffle(pickList)
    #print pickList
    for idx in pickList:
        print "----- Seq No.:",idx+1," (iter:",train_num+1,")-----"
        x = xs[idx]
        #print x 
        y_hat = y_hats[idx]
        V = Viterbi (x , w , y_class , y_hat , 0)
        start = time.clock()
        y_tilde = V.main_Viterbi()
        end = time.clock()
        print "Viterbi time :" , end-start
        print "tilde " , np.dot(w.T , Psi( x , y_tilde ))
        print "hat   " , np.dot(w.T , Psi( x , y_hat) )
        
        if np.dot(w.T , Psi( x , y_tilde )) < np.dot(w.T , Psi( x , y_hat) ) :
            print "there is something error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print "there is something error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print "there is something error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print "there is something error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print "there is something error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        w = update_w( w , x , y_hat , y_tilde , V)
        #print "w  ",w
        print "after update tilde " , np.dot(w.T , Psi( x , y_tilde )) 
        print "after update hat   " , np.dot(w.T , Psi( x , y_hat) )   
        print "===================================================================="
        if train_num==9:
            print x
            print y_tilde
            print y_hat
    #========= pickle the model========
    #if (train_num+1)%10 == 0:
    print "Saving model at:",train_num+1
    filehandler = open(MODEL_ROOT+MODEL_NAME,'wb')
    saved_params = (w)
    cPickle.dump(saved_params, filehandler)
    filehandler.close()
    
    if np.array_equal(temp_w,w):
        break


#========= test =========
def load_liststateto48():
    liststateto48 = []
    with open('Data/phones/state_48_39.map','r') as fin:
        for line in fin:
            liststateto48.append(line.split()[1])
    return liststateto48

def load_dict_48to39():
    d = dict()
    with open('Data/phones/48_39.map') as fin:
        for row in fin:
            pair = row.split()
            d[ pair[0] ] = pair[1]
    
    if len(set(d.keys())) != 48 or len(set(d.values())) != 39:
        print "Error phoneme amount!"
    return d

def load_dict_48toChr():
    d = dict()
    with open('Data/phones/48_idx_chr.map') as fin:
        for line in fin:
            ph48, idx, char, nocare= line.split()
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
    smooth_y.append(y_temp[0])
    for i in xrange(1,len(y_temp)-1):
        if y_temp[i-1] == y_temp[i+1] and \
           y_temp[i]   != y_temp[i-1]:
            smooth_y.append( y_temp[i-1] )
        elif y_temp[i] != y_temp[i-1] and \
             y_temp[i] != y_temp[i+1]:
            # discard
            continue
            #final_seq.append( y_temp[i-1] )
        else:
            smooth_y.append( y_temp[i] )
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


print "finish training ++++++++++++++++++++++++++++"
'''print x
print y_tilde 
print y_hat '''
