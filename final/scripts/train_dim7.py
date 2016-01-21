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

import Keras_model


BATCH_SIZE = 1
MAX_EPOCH = 20
SAVE_EPOCH = 1
VAL_RATIO = 0.2
MODEL_FILE_NAME = '../models/1.10_dim7'
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
pickList = range(0,len(ID_PKL))
random.shuffle(pickList)
val_pickList = []
for idx in xrange(0,int(len(pickList)*VAL_RATIO)):
    val_pickList.append( pickList.pop(idx) )
# maybe this will help?  -- Ray.
# this really help, thanks!  -- Angus.
#print '{0:{fill}{align}12}'.format(ID_PKL[0][0],fill='0',align='>')

# printing 300-dim word vector
#print word_vec[ QUESTION_PKL[0][0] ]
print "start making model..."
model = Keras_model.keras_model(20)

#===== Start training =====
print "Start training!"
for epoch in xrange(MAX_EPOCH):
    tStart = time.time()
    print "Epoch:",epoch+1
    progress = ProgressBar().start()
    cost = 0
    errorsum = 0
    for batch_num in xrange(0,int(len(pickList)/BATCH_SIZE)):
        progress.update(int((batch_num+1)/float(146962*(1-VAL_RATIO))*100))
        start = batch_num*BATCH_SIZE    
        end   = (batch_num+1)*BATCH_SIZE
        if end > len(pickList):
            start = len(pickList)-BATCH_SIZE
            end = len(pickList)
        batched_image = []
        batched_words = []
        batched_opt1 = []
        batched_opt2 = []
        batched_opt3 = []
        batched_opt4 = []
        batched_opt5 = []
        batched_sol  = []
        for idx in pickList[start:end]:
            wordsList = []
            for word in QUESTION_PKL[idx]:
                if word in word_vec:
                    wordsList.append(word_vec[word])
            words = np.vstack(wordsList)
            
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
                        batched_opt1.append( np.vstack(opt) )
                elif opt_idx == 1:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt2.append( np.vstack(opt) )
                elif opt_idx == 2:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt3.append( np.vstack(opt) )
                elif opt_idx == 3:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt4.append( np.vstack(opt) )
                elif opt_idx == 4:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt5.append( np.vstack(opt) )
            #Image attached
            batched_image.append(mem_image[ IM_ID_DICT[ '{0:{fill}{align}12}'.format(ID_PKL[idx][0],fill='0',align='>') ] ].flatten())
            #batched_words.append([words])
            batched_words.append(words)
        image_input = np.swapaxes( \
                        np.swapaxes( \
                            #np.asarray(batched_image),1,2
                            np.dstack(batched_image),1,2
                        ),0,1
                      )
        question_input = np.swapaxes( \
                            np.swapaxes( \
                                np.dstack(batched_words),1,2
                            ),0,1
                         )
        ans1_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt1),1,2
                        ),0,1
                     )
        ans2_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt2),1,2
                        ),0,1
                     )
        ans3_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt3),1,2
                        ),0,1
                     )
        ans4_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt4),1,2
                        ),0,1
                     )
        ans5_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt5),1,2
                        ),0,1
                     )
        solution   = np.vstack(batched_sol)
        #pdb.set_trace()
        image_new = np.array([image_input[0][0]])
        cost += model.train_on_batch([question_input , ans1_input 
                                  , ans2_input
                                  , ans3_input
                                  , ans4_input
                                  , ans5_input , image_new ] , solution )[0]
    
    print "Total cost =",cost
    tEnd = time.time()
    print "Time used:", tEnd-tStart
    if (epoch+1)%SAVE_EPOCH==0:
        model.save_weights(MODEL_FILE_NAME +'_' +str(epoch+1)+'.hdf5' )
        print "       model saved to",MODEL_FILE_NAME+str(epoch+1) 
    #===== validation set =====
    for batch_num in xrange(0,int(len(val_pickList)/BATCH_SIZE)):
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
            words = np.vstack(wordsList)
            
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
                        batched_opt1.append( np.vstack(opt) )
                elif opt_idx == 1:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt2.append( np.vstack(opt) )
                elif opt_idx == 2:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt3.append( np.vstack(opt) )
                elif opt_idx == 3:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt4.append( np.vstack(opt) )
                elif opt_idx == 4:
                    if not opt:
                        pdb.set_trace()
                    else:
                        batched_opt5.append( np.vstack(opt) )
            #Image attached
            batched_image.append(mem_image[ IM_ID_DICT[ '{0:{fill}{align}12}'.format(ID_PKL[idx][0],fill='0',align='>') ] ].flatten())
            #batched_words.append([words])
            batched_words.append(words)
        image_input = np.swapaxes( \
                        np.swapaxes( \
                            #np.asarray(batched_image),1,2
                            np.dstack(batched_image),1,2
                        ),0,1
                      )
        question_input = np.swapaxes( \
                            np.swapaxes( \
                                np.dstack(batched_words),1,2
                            ),0,1
                         )
        ans1_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt1),1,2
                        ),0,1
                     )
        ans2_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt2),1,2
                        ),0,1
                     )
        ans3_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt3),1,2
                        ),0,1
                     )
        ans4_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt4),1,2
                        ),0,1
                     )
        ans5_input = np.swapaxes( \
                        np.swapaxes( \
                            np.dstack(batched_opt5),1,2
                        ),0,1
                     )
        solution   = np.vstack(batched_sol)
        #pdb.set_trace()
        image_new = np.array([image_input[0][0]])
        prediction = model.predict_on_batch([question_input , ans1_input 
                                  , ans2_input
                                  , ans3_input
                                  , ans4_input
                                  , ans5_input , image_new ])
        answer = np.argmax(solution,axis=1)
        guess = np.argmax(prediction[0],axis=1)
        error = np.sum(answer != guess)
        errorsum += error/float(BATCH_SIZE)
    error_rate = errorsum/float(int(len(pickList)/BATCH_SIZE)-int(len(pickList)*(1-VAL_RATIO)/BATCH_SIZE))
    print "Validation Error Rate:",error_rate*100,"%"


    random.shuffle(pickList)

print "Finished training!"


