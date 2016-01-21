import random
import time
import theano
import theano.tensor as T
import numpy as np
import cPickle
import pdb
from itertools import izip

from phonemap import *


# Output Format: single list
def readfile_for_test(filename,ratio=0.8):
    print "Reading data..."
    inputs = []
    names = []
    frame_total = []
    frame_max = []
    num=0
    length=0
    with open (filename,'r') as f:
        for line in f: 
            words = line.strip('\n').split()
            if int(words[0].split('_')[2])==1 and num !=0:
                frame_total.append(num)
                frame_max+=num*[[num]]
                num=1
            else:
                num+=1
            names.append(words[0])
            inputs.append([ float(num_) for num_ in words[1:] ])
            length+=1
    f.close()
    frame_total.append(num)
    frame_max+=num*[[num]]

    print "Normalizing..."
    vector_len = len(inputs[0])
    for idx in range(0,vector_len):
        maxi = max(vector[idx] for vector in inputs)
        mini = min(vector[idx] for vector in inputs)
        center = (maxi+mini)/2
        span = (maxi-mini)/2
        for vector in inputs:
            vector[idx] = ( vector[idx]-center )/span

    print "Patching..."
    new_input = []
    kind = 0
    for idx in range(0,len(inputs)):
        data = inputs[idx]
        frame_num = int(names[idx].split('_')[2])
        themax = frame_total[kind]
        if frame_num>4 and (themax-frame_num)>3:
            new_sub = inputs[idx-4]+inputs[idx-3]+inputs[idx-2]+inputs[idx-1]+data \
                    +inputs[idx+1]+inputs[idx+2]+inputs[idx+3]+inputs[idx+4]
        else:
            if frame_num == 1:
                new_sub = data*5+inputs[idx+1]+inputs[idx+2]+inputs[idx+3]+inputs[idx+4]
            elif frame_num == 2:
                new_sub = inputs[idx-1]*4+data+inputs[idx+1]+inputs[idx+2]+inputs[idx+3]+inputs[idx+4]
            elif frame_num == 3:
                new_sub = inputs[idx-2]*3+inputs[idx-1]\
                        +data+inputs[idx+1]+inputs[idx+2]+inputs[idx+3]+inputs[idx+4]
            elif frame_num == 4:
                new_sub = inputs[idx-3]*2+inputs[idx-2]+inputs[idx-1]\
                        +data+inputs[idx+1]+inputs[idx+2]+inputs[idx+3]+inputs[idx+4]
            elif (themax-frame_num) == 0:
                new_sub = inputs[idx-4]+inputs[idx-3]+inputs[idx-2]+inputs[idx-1]+data*5
            elif (themax-frame_num) == 1:
                new_sub = inputs[idx-4]+inputs[idx-3]+inputs[idx-2]+inputs[idx-1]+data+inputs[idx+1]*4
            elif (themax-frame_num) == 2:
                new_sub = inputs[idx-4]+inputs[idx-3]+inputs[idx-2]+inputs[idx-1]+data\
                        +inputs[idx+1]+inputs[idx+2]*3
            elif (themax-frame_num) == 3:
                new_sub = inputs[idx-4]+inputs[idx-3]+inputs[idx-2]+inputs[idx-1]+data\
                        +inputs[idx+1]+inputs[idx+2]+inputs[idx+3]*2
        new_input.append(new_sub)
        if frame_num == themax:
            kind += 1
    del inputs
    del frame_total

    return names,new_input[:int(length*ratio)],new_input[int(length*ratio):]

def shuffle(pickList):
    random.seed(time.time())
    random.shuffle(pickList)
    return pickList

def parse_val_set(mem_data,mem_label,pickList,IDs,ratio):
    val_set = []
    val_lab = []
    val_IDs = []
    length = len(pickList)
    if ratio != 1:
        pivot = 0
        POP = [pickList.pop(num) for num in range(int(length*(1-ratio)))]
        val_IDs = [IDs[pop] for pop in POP]
        val_set = np.vstack(mem_data[:,POP])
        #val_lab = np.hstack(mem_label[POP])
        #pdb.set_trace()
        #val_set = mem_data[:,POP]
        val_lab = mem_label[POP]

    return pickList,val_set,val_lab,val_IDs

def readID(PKL_ID):
    with open(PKL_ID,"rb") as filehandler:
        IDs = cPickle.load(filehandler)
    filehandler.close()
    return IDs
