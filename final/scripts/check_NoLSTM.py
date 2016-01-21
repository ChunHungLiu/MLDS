import cPickle as pickle
import numpy as np
import pdb
import random
import time
#import pyprind
from progressbar import ProgressBar
# load word vector
from gensim.models.word2vec import Word2Vec
wordvec_file = '../Data/glove.pruned.300d.txt'
word_vec = Word2Vec.load_word2vec_format(wordvec_file, binary=False)

#import Keras_model_noLSTM
#import Keras_model_deep_noLSTM
import Keras_model_attention_2

BATCH_SIZE = 197
MAX_EPOCH = 200
SAVE_EPOCH = 20
VAL_RATIO = 0.0
MODEL_FILE_NAME = '../models/attention2_1_8_120.hdf5'
#===== load training data =====
data_prefix = '../Data/pkl/'
paths = [
    'img_q_id_train',
    'question_train',
    'choices_train',
    'answer_train',
    'answer_train_sol',
    'annotation_train',
    #'img_q_id_test',
    #'question_test',
    #'choices_test'
]
#load wrods 
ID_PKL = pickle.load(open(data_prefix+paths[0]+'.pkl','rb'))
QUESTION_PKL = pickle.load(open(data_prefix+paths[1]+'.pkl','rb'))
CHOICE_PKL = pickle.load(open(data_prefix+paths[2]+'.pkl','rb'))
ANSWER_PKL = pickle.load(open(data_prefix+paths[3]+'.pkl','rb'))
ANS_SOL_PKL = pickle.load(open(data_prefix+paths[4]+'.pkl','rb'))
ANNO_PKL = pickle.load(open(data_prefix+paths[5]+'.pkl','rb'))
#load picture features
IM_ID = pickle.load(open('../Data/train2014/ID.pkl','rb'))
IM_ID_DICT = dict()
for num in xrange(len(IM_ID)):
    ID = IM_ID[num].split('_')[2].split('.')[0]
    IM_ID_DICT[ID]=num
mem_shape = (82783,1,1000)
mem_image = np.memmap('../Data/train2014/vgg_feats.memmap',dtype='float32',mode='r',shape=mem_shape )
#===== prepare pickList =====
val_pickList = range(0,len(ID_PKL))
# maybe this will help?  -- Ray.
# this really help, thanks!  -- Angus.
#print '{0:{fill}{align}12}'.format(ID_PKL[0][0],fill='0',align='>')

# printing 300-dim word vector
#print word_vec[ QUESTION_PKL[0][0] ]
print "start making model..."
model = Keras_model_attention_2.keras_model(1)
#model = Keras_model_noLSTM.keras_model(1)
model.load_weights(MODEL_FILE_NAME)
for epoch in xrange(1):#NULL loop
    errorsum = 0
    progress = ProgressBar().start()
    for batch_num in xrange(0,int(len(val_pickList)/BATCH_SIZE)):
        progress.update(int((batch_num+1)/float(146962/BATCH_SIZE)*100))
        start = batch_num*BATCH_SIZE    
        end   = (batch_num+1)*BATCH_SIZE
        if end > len(val_pickList):
            start = len(val_pickList)-BATCH_SIZE
            end = len(val_pickList)
        batched_image = []
        batched_words = []
        batched_opt1 = []
        batched_opt2 = []
        batched_opt3 = []
        batched_opt4 = []
        batched_opt5 = []
        batched_sol  = []
        for idx in val_pickList[start:end]:
            wordsList = []
            for word in QUESTION_PKL[idx]:
                if word in word_vec:
                    wordsList.append(word_vec[word])
            words = np.mean(wordsList, axis=0)
            batched_words.append(words)
            
            sol = ANS_SOL_PKL[idx]
            sol_vec = np.zeros(5)
            sol_vec[sol] = 1.0
            batched_sol.append(sol_vec)
            #batched_opt1 = np.hstack([ word_vec[word] if word in word_vec else np.zeros((300,)) for word in CHOICE_PKL[idx][0] ] )
            for opt_idx,choice in enumerate(CHOICE_PKL[idx]):
                opt=[]
                for word in choice:
                    if word in word_vec:
                        opt.append(word_vec[word])
                    else:
                        opt.append(np.zeros((300,)) )
                if not opt:
                    opt.append(np.zeros((300,)) )
                # no switch in python :P
                if   opt_idx == 0:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt1.append( np.mean(opt, axis=0) )
                elif opt_idx == 1:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt2.append( np.mean(opt, axis=0) )
                elif opt_idx == 2:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt3.append( np.mean(opt, axis=0) )
                elif opt_idx == 3:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt4.append( np.mean(opt, axis=0) )
                elif opt_idx == 4:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt5.append( np.mean(opt, axis=0) )
            #Image attached
            batched_image.append(mem_image[ IM_ID_DICT[ '{0:{fill}{align}12}'.format(ID_PKL[idx][0],fill='0',align='>') ] ].flatten())
        image_input     = np.vstack(batched_image)
        question_input  = np.vstack(batched_words)
        ans1_input      = np.vstack(batched_opt1)
        ans2_input      = np.vstack(batched_opt2)
        ans3_input      = np.vstack(batched_opt3)
        ans4_input      = np.vstack(batched_opt4)
        ans5_input      = np.vstack(batched_opt5)
        solution        = np.vstack(batched_sol)
        data = np.hstack([image_input,question_input,ans1_input,ans2_input,ans3_input,ans4_input,ans5_input])
        prediction = model.predict_on_batch( data)
        answer = np.argmax(solution,axis=1)
        guess = np.argmax(prediction[0],axis=1)
        error = np.sum(answer != guess)
        errorsum += error/float(BATCH_SIZE)

    error_rate = errorsum/float(int(len(val_pickList)/BATCH_SIZE))
    print "Validation Error Rate:", error_rate*100,"%"

print "Finished training!"


