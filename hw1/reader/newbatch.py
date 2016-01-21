import numpy as np
import random
import time

from newphone import *

def readData(filename):
    input_x = []
    with open(filename,'r') as fin:
        for line in fin:
            words = line.strip('\n').split()
            line_x = [ words[0] ]+[ [ float(num) for num in words[1:] ] ]
            input_x.append(line_x)
    #  [ [ ID, Feature vector ], [...],...,[...] ]
    return input_x

def readLabel(filename):
    label = dict()
    with open (filename,'r') as fin:
        for line in fin:
            parsed = line.strip('\n').split(',')
            label[ parsed[0] ] = parsed[1]
    return label

def splitset(dataset,ratio = 0.8):
    random.seed(10)
    random.shuffle(dataset)
    div = int(len(dataset)*ratio)
    return dataset[:div],dataset[div:]

def sepIDnFeat(dataset):
    IDs = []
    Feats = []
    for data in dataset:
        IDs.append(data[0])
        Feats.append(data[1])
    return IDs,Feats

def label( src,src2tar_dict):
    return [ src2tar_dict[ID] for ID in src ]

def index2vector(indices,phonenum = 48):
    vecs = []
    for idx in indices:
        v = [0] * phonenum
        v[idx] = 1
        vecs.append(v)
    return vecs

def batch(dataset,labelset,batch_size):
    assert(len(dataset) == len(labelset))
    random.seed(time.time())

    total = zip(dataset,labelset)
    random.shuffle(total)
    dataset = list(zip(*total))[0]
    labelset = list(zip(*total))[1]

    num = len(dataset)

    b_dataset  = [  dataset[i:i+batch_size] for i in range(0,num,batch_size)]
    b_labelset = [ labelset[i:i+batch_size] for i in range(0,num,batch_size)]

    return b_dataset,b_labelset


def batchtoNpCol(batches):
    ret = []
    for batch in batches:
        arr = np.asarray(batch,dtype='float32')
        ret.append(np.transpose(arr))
    return ret
