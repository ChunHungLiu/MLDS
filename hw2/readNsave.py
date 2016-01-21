import cPickle as pickle
import time
import pdb
import random

from itertools import izip
import numpy as np

tStart = time.time()

TRAIN_FILENAME = "./Data/fbank/train.ark"
LABEL_FILENAME = "./Data/label/train.lab"###reading 48 for SVM!!!!
MEM_ID = 'ID.pkl'
MEM_DATA = 'data.fbank.memmap'
MEM_LABEL = 'label48.memmap'
mapdir2 = './Data/phones/state_48_39.map'

def load_dict_IdxPh48():
    d = dict()
    with open(mapdir2,'r') as fin:
        count = 0
        for line in fin:
            if count >= 48:
                break
            _, ph48, _ = line.split()
            d[ph48] = count
            count += 1
    return d
to48 = load_dict_IdxPh48()
def readLabel():
    label = dict()
    with open (LABEL_FILENAME,'r') as fin:
        for line in fin:
            parsed = line.strip('\n').split(',')
            label[ parsed[0] ] = to48[parsed[1]]
    fin.close()
    
    return label

print "Reading data..."
inputs = []
names = []
frame_total = []
frame_max = []
num=0
length=0
with open (TRAIN_FILENAME,'r') as f:
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
# IDs = ['name',total_pos,index to memmap,frame_batch begin position,frame max num]
#         hence begin position+frame max num = end
# DATAs = [[1st frame_batch(numpy arrays)],...,[last frame_batch]] where frame_batch = every frames with same title

print "Read and matching label feature vector..."
label_dict = readLabel()
label = []
for name in names:
    label.append(label_dict[name])
del label_dict

print "Transforming data into numpy..."
DATAs = np.asarray(new_input,dtype='float32').T
del new_input
LABELs = np.asarray(label,dtype='float32')
del label
pdb.set_trace()
##### adding frame_max
IDs = [[n]+f for n,f in zip(names,frame_max)]
del frame_max
#######################
##   Serialization   ##
#######################
#Please transfer every frame_batch in DATAs into a memmep
'''
IDs [str] ->lists
DATAs ->numpys
LABELs ->lists
'''
fh = open(MEM_ID,'wb')
pickle.dump(IDs, fh, pickle.HIGHEST_PROTOCOL)
fh.close()
print "Saved IDs to \"{0}\"...".format(MEM_ID)

#pdb.set_trace()
mem_shape = DATAs.shape
mem_data = np.memmap(MEM_DATA,dtype='float32',mode='w+',shape=mem_shape)
mem_data[:] = DATAs[:]
print "Saved DATAs to \"{0}\"...".format(MEM_DATA)
print "mem_data shape:",mem_shape

mem_label = np.memmap(MEM_LABEL,dtype='int16',mode='w+',shape=LABELs.shape)
mem_label[:] = LABELs[:]
print "Saved LABELs to \"{0}\"...".format(MEM_LABEL)
print "mem_label shape:",LABELs.shape

''' example
read a saved memmap:

<variable> = \
np.memmap('<filename>',
          dtype='float32',
          mode='r', (read-only)
          shape=<the_same_shape_as_saving> )

'''

print "Done transfering frame_batch in \'{0}\'.".format(MEM_DATA)
#######################

tEnd = time.time()
print "Total_length:",length
print "Data prepared time:",tEnd-tStart
