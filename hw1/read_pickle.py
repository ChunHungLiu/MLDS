import cPickle
import random
import time

TRAIN_FILENAME = "./Data/fbank/train.ark"
LABEL_FILENAME = "./Data/state_label/train.lab"

ratio = 0.99

tStart = time.time()

print "Reading data..."
input_x = []
frame_total = []
num=0
with open (TRAIN_FILENAME,'r') as f:
    for line in f: 
        words = line.strip('\n').split()
        if int(words[0].split('_')[2])==1 and num !=0:
            frame_total.append(num)
            num=1
        else:
            num+=1
        line_x = [ words[0] ]+[ [ float(num_) for num_ in words[1:] ] ]
        input_x.append(line_x)
f.close()
frame_total.append(num)

print "Normalizing..."
vector_len = len(input_x[0][1])
for idx in range(0,vector_len):
    maxi = max(vector[1][idx] for vector in input_x)
    mini = min(vector[1][idx] for vector in input_x)
    center = (maxi+mini)/2
    span = (maxi-mini)/2
    for vector in input_x:
        vector[1][idx] = ( vector[1][idx]-center )/span
'''
print "Patching..."
new_input = []
kind = 0
for idx in range(0,len(input_x)):
    data = input_x[idx]
    frame_num = int(data[0].split('_')[2])
    themax = frame_total[kind]
    if frame_num>4 and (themax-frame_num)>3:
        new_sub = input_x[idx-4][1]+input_x[idx-3][1]+input_x[idx-2][1]+input_x[idx-1][1]+data[1] \
                +input_x[idx+1][1]+input_x[idx+2][1]+input_x[idx+3][1]+input_x[idx+4][1]
    else:
        if frame_num == 1:
            new_sub = data[1]*5+input_x[idx+1][1]+input_x[idx+2][1]+input_x[idx+3][1]+input_x[idx+4][1]
        elif frame_num == 2:
            new_sub = input_x[idx-1][1]*4\
                    +data[1]+input_x[idx+1][1]+input_x[idx+2][1]+input_x[idx+3][1]+input_x[idx+4][1]
        elif frame_num == 3:
            new_sub = input_x[idx-2][1]*3+input_x[idx-1][1]\
                    +data[1]+input_x[idx+1][1]+input_x[idx+2][1]+input_x[idx+3][1]+input_x[idx+4][1]
        elif frame_num == 4:
            new_sub = input_x[idx-3][1]*2+input_x[idx-2][1]+input_x[idx-1][1]\
                    +data[1]+input_x[idx+1][1]+input_x[idx+2][1]+input_x[idx+3][1]+input_x[idx+4][1]
        elif (themax-frame_num) == 0:
            new_sub = input_x[idx-4][1]+input_x[idx-3][1]+input_x[idx-2][1]+input_x[idx-1][1]+data[1]*5
        elif (themax-frame_num) == 1:
            new_sub = input_x[idx-4][1]+input_x[idx-3][1]+input_x[idx-2][1]+input_x[idx-1][1]+data[1]\
                    +input_x[idx+1][1]*4
        elif (themax-frame_num) == 2:
            new_sub = input_x[idx-4][1]+input_x[idx-3][1]+input_x[idx-2][1]+input_x[idx-1][1]+data[1]\
                    +input_x[idx+1][1]+input_x[idx+2][1]*3
        elif (themax-frame_num) == 3:
            new_sub = input_x[idx-4][1]+input_x[idx-3][1]+input_x[idx-2][1]+input_x[idx-1][1]+data[1]\
                    +input_x[idx+1][1]+input_x[idx+2][1]+input_x[idx+3][1]*2
    new_input.append([data[0],new_sub])
    if frame_num == themax:
        kind += 1

del input_x
'''
del frame_total
if ratio != 1:
    random.seed(1)
    random.shuffle(input_x)
total_len = len(input_x)

#labeled_training_set = [TRAINING_DATA]
print "cPickle data..."

with open("lab_train.p","wb") as filehandler:
	cPickle.dump(input_x[:int(total_len*ratio)],filehandler,cPickle.HIGHEST_PROTOCOL)
filehandler.close()

with open("lab_val.p","wb") as filehandler:
	cPickle.dump(input_x[int(total_len*ratio):],filehandler,cPickle.HIGHEST_PROTOCOL)
filehandler.close()


############read label#########
print "Reading label..."

label = dict()
with open (LABEL_FILENAME,'r') as fin:
	for line in fin:
		parsed = line.strip('\n').split(',')
		label[ parsed[0] ] = parsed[1]
fin.close()
print "cPickle label..."
with open("label.p","wb") as filehandler:
	cPickle.dump(label,filehandler,cPickle.HIGHEST_PROTOCOL)
filehandler.close()

tEnd = time.time()
print "time:",tEnd-tStart
