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


def readfile_(filename,ratio=0.8):
    input_x = []
    with open (filename,'r') as f:
        for line in f:
            words = line.strip('\n').split()
            line_x = [ words[0] ]+[ [ float(num) for num in words[1:] ] ]
            input_x.append(line_x)
    f.close()

    print "Normalizing..."
    vector_len = len(input_x[0][1])
    for idx in range(0,vector_len):
        maxi = max(vector[1][idx] for vector in input_x)
        mini = min(vector[1][idx] for vector in input_x)
        center = (maxi+mini)/2
        span = (maxi-mini)/2
        for vector in input_x:
            vector[1][idx] = ( vector[1][idx]-center )/span

    # Separates Train and Validation Set Randomly
    num = len(input_x)
    if ratio != 1:
        random.seed(10)
        random.shuffle(input_x)

    return input_x[:int(num*ratio)],input_x[int(num*ratio):]


def readfile():
    '''
    Return Train and Validation sets
    Ratio: Train / Total
    '''
    with open("lab_train.p","rb") as filehandler:
        labeled_training_set = cPickle.load(filehandler)

    with open("lab_val.p","rb") as filehandler:
        labeled_val_set = cPickle.load(filehandler)
    filehandler.close()

    return labeled_training_set,labeled_val_set

def readfile_inloop():
    with open("lab_train.p","rb") as filehandler:
        labeled_training_set = cPickle.load(filehandler)
    filehandler.close()

    return labeled_training_set

def removeLabel(LabeledData):
    # Data Format: [ "label", [ Feature vector ] ]
    return [ data[1] for data in LabeledData ]

def removeBatchLabel(LabeledBatches):
    # Data Format: [ [ ["label",[Feature vector]],[],..,[] ], [ batch ],...,[ batch ] ]
    # Expected Format: [ [ [Feat],[Feat],...,[Feat] ],[Batch],..,[Batch]]
    unlabeledbatches = []
    for idx,batch in enumerate(LabeledBatches):
        unlabeledbatches.append(removeLabel(batch))
    return unlabeledbatches



def batch(data,batch_size=10):
    random.seed(time.time())
    random.shuffle(data)

    num = len(data)

    if num % batch_size != 0:
        c = batch_size - num % batch_size
        data = data + data[:c]

    batches = [ data[i:i+batch_size] for i in range(0,num,batch_size)]

    return batches

def readLabel():
    with open("label.p","rb") as filehandler:
        label = cPickle.load(filehandler)
    filehandler.close()
    
    return label

def MatchLabel2Batches(batches,label):
    batched_label = []
    for batch in batches:
        batched_label.append( [ (x[0],label[x[0] ]) for x in batch ] )

    return batched_label

def BatchedLabelToVector(labeledbatches):
    vectorbatches = []
    #p = get_PhoneDict(48)
    p = get_PhoneStateDict()
    for batch in labeledbatches:
        vectorbatches.append([ p[lab] for lab in batch])
    del p
    return vectorbatches


def BatchToNPCol(batches):
    ret = []
    for batch in batches:
        arr = np.asarray(batch,dtype='float32')
        ret.append(np.transpose(arr))
    return ret

def SepIDnFeat(TEST_DATA):
    IDs = []
    Feats = []
    for data in TEST_DATA:
        IDs.append(data[0])
        Feats.append(data[1])
    return IDs,Feats
