from itertools import izip
import time 
import pdb
import numpy as np
import mnist_loader
from nnet import *
import cPickle as pickle
import shelve

#dnn = DNN([784,1000,1000,1000,10])
dnn = DNN([784,100,10])


corr = [0.1,0.2,0.3]
DAs = []
for idx in xrange(1, len(dnn.layers)-1):
    DAs.append( DA(dnn.layers[idx],
                   dnn.W[idx],
                   dnn.b[idx],
                   ) )

print "start reading"
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()

MAX_EPOCH  = 100
PRETRAIN_EPOCH = 0
BATCH_SIZE = 100
PRETRAIN_BATCH_SIZE=100

data = np.concatenate([ d for d,l in training_data], axis=1).astype('float32')
label = np.concatenate([ l for d,l in training_data], axis=1).astype('float32')


print "Start pre-training. pretrain {0} epoches".format(PRETRAIN_EPOCH)
prop_input = data
for l,da in enumerate(DAs):
    for epoch in xrange(PRETRAIN_EPOCH):
        batch_cost = 0
        tStart = time.time()
        for i in xrange( (data.shape[1]-1)/PRETRAIN_BATCH_SIZE + 1):
            begin = i * PRETRAIN_BATCH_SIZE
            if (i+1)*PRETRAIN_BATCH_SIZE > data.shape[1]:
                end = data.shape[1]
            else:
                end = (i+1)*PRETRAIN_BATCH_SIZE
            b = da.train(prop_input[:,begin:end])
            if b != b:
                pdb.set_trace()
            batch_cost += b
        tEnd = time.time()
        print "Layer:{0}, Epoch:{1}, cost:{2}, time:{3}".format(
            l,epoch+1,batch_cost,tEnd-tStart)
    prop_input = da.get_hidden(prop_input)


print "Start training. train {0} epoches".format(MAX_EPOCH)
for epoch in xrange(MAX_EPOCH):
    tStart = time.time()
    cost = 0
    for i in xrange( (len(training_data)-1)/BATCH_SIZE + 1):
        begin = i * BATCH_SIZE
        if (i+1)*BATCH_SIZE > data.shape[1]:
            end = data.shape[1]
        else:
            end = (i+1)*BATCH_SIZE
        cost += dnn.train(data[:,begin:end], label[:,begin:end],
                          0.1,
                          0.9,
                          1)
        #print "BATCH {0}, {1} cost {2}".format(begin,end, c)
    tEnd = time.time()
    print "Epoch {0}, Cost {1}, Epoch time {2}".format(epoch+1, cost, tEnd-tStart)

t_data = np.concatenate([ d for d,l in test_data], axis=1).astype('float32')
t_label = [l for d,l in test_data]

# test saving
fh = open('test_savemodel.tmp','wb')
pickle.dump((dnn.layers, dnn.W, dnn.b),fh)
fh.close()
del dnn
fh = open('test_savemodel.tmp','rb')
l,ws,bs = pickle.load(fh)
loaded_dnn = DNN(l,ws,bs)
fh.close()

result = loaded_dnn.test(t_data)
result = [r.argmax() for r in np.split(result,result.shape[1],axis=1)]
correct = 0
for r,a in izip(result, t_label):
    if r == a:
        correct += 1

print "Correctness:{0}".format(float(correct)/len(t_label))


